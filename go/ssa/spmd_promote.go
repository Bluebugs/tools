// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/constant"
	"go/token"
	"go/types"
)

// arrayCopyInfo captures the external-initialization pattern for an array:
// a Slice of the alloc fed into a builtin copy call, plus the load
// instructions that read back through IndexAddr or via a whole-array load.
//
// Two access sub-patterns are supported:
//
//  1. IndexAddr pattern (rangeint / plain index):
//     t1 = slice alloc[:]
//     t2 = copy(t1, src)
//     t3 = &alloc[i]        // IndexAddr
//     t4 = *t3              // UnOp{MUL} load — collected in loads
//
//  2. Array-value pattern (rangeindex over array):
//     t1 = slice alloc[:]
//     t2 = copy(t1, src)
//     t3 = *alloc           // UnOp{MUL} whole-array load — arrayLoad
//     t4 = t3[i]            // Index on array value — collected in indexInstrs
type arrayCopyInfo struct {
	sliceInstr  *Slice   // t1 = slice alloc[:]
	copyCall    *Call    // t2 = copy(t1, src)
	srcArg      Value    // second argument to copy (string or []byte slice)
	loads       []*UnOp  // UnOp{MUL} load results through IndexAddr (pattern 1)
	arrayLoad   *UnOp    // UnOp{MUL} whole-array load from alloc (pattern 2)
	indexInstrs []*Index // Index instructions on arrayLoad result (pattern 2)
}

// spmdLoopBodyBlocks returns all basic blocks that belong to the body of an
// SPMD loop. This includes BodyBlock, LoopBlock, and any intermediate blocks
// created by varying control flow (if/else, switch) within the body.
//
// The algorithm is a BFS from BodyBlock that stops at DoneBlock (loop exit)
// and EntryBlock (loop entry predecessor). For non-merged loops (rangeindex),
// the walk stops at LoopBlock without recursing into its successors, since
// LoopBlock's successors are the back-edge to BodyBlock and DoneBlock — both
// of which are already handled.
func spmdLoopBodyBlocks(loop *SPMDLoopInfo) []*BasicBlock {
	if loop.BodyBlock == nil {
		return nil
	}
	visited := make(map[*BasicBlock]bool)
	var blocks []*BasicBlock
	var walk func(b *BasicBlock)
	walk = func(b *BasicBlock) {
		if b == nil || visited[b] {
			return
		}
		// DoneBlock is outside the loop — stop here.
		if b == loop.DoneBlock {
			return
		}
		// EntryBlock is before the loop — stop here.
		if b == loop.EntryBlock {
			return
		}
		visited[b] = true
		blocks = append(blocks, b)
		// For non-merged loops (rangeindex), LoopBlock is the loop header whose
		// successors are BodyBlock (back-edge) and DoneBlock (exit). Including
		// LoopBlock itself is correct, but we must not recurse past it to avoid
		// re-visiting BodyBlock and collecting DoneBlock.
		if b == loop.LoopBlock && !loop.MergedBodyLoop {
			return
		}
		for _, succ := range b.Succs {
			walk(succ)
		}
	}
	walk(loop.BodyBlock)
	// For non-merged loops, ensure LoopBlock is included even if the BFS did
	// not reach it via BodyBlock's successors (shouldn't happen, but be safe).
	if loop.LoopBlock != nil && !visited[loop.LoopBlock] {
		blocks = append(blocks, loop.LoopBlock)
	}
	return blocks
}

// promoteSPMDArrays promotes eligible small array allocations inside SPMD
// loops. An Alloc is eligible when:
//
//   - Type is *[N]T where N*sizeof(T) <= 16 (fits in v128)
//   - The loop BoundValue is a constant equal to N (array length == loop bound)
//   - All referrers are IndexAddr with index == unwrapChangeType(IterPhi)
//   - Either all IndexAddr referrers are Store (as Addr) [write-only pattern]
//   - Or the alloc also has a Slice→copy referrer and all IndexAddr referrers
//     are UnOp{MUL} loads [copy+read pattern]
//   - All use sites are inside the enclosing SPMD loop body
//
// This pass runs after resolveSPMDLoops (so IterPhi is available) and before
// predicateSPMD (so stores are still regular *Store instructions from the
// builder, not yet converted to SPMDStore by spmdConvertLoopOps). Removing
// write-only array allocations here prevents TinyGo from generating unnecessary
// per-lane GEP + scatter sequences for simple lane-indexed writes.
//
// For the copy+read pattern, an SPMDVectorFromMemory instruction is emitted
// to load the external data directly into a varying value, replacing the
// copy + array reads with a single vector load.
func promoteSPMDArrays(f *Function) {
	if len(f.SPMDLoops) == 0 {
		return
	}

	// Build a map from block to its enclosing SPMDLoopInfo.
	// Use spmdLoopBodyBlocks to cover all blocks inside the loop body,
	// including intermediate blocks created by varying control flow.
	blockToLoop := make(map[*BasicBlock]*SPMDLoopInfo)
	for _, loop := range f.SPMDLoops {
		for _, b := range spmdLoopBodyBlocks(loop) {
			blockToLoop[b] = loop
		}
	}

	type candidate struct {
		alloc    *Alloc
		loop     *SPMDLoopInfo
		copyInfo *arrayCopyInfo // nil for write-only pattern
	}
	var candidates []candidate

	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			alloc, ok := instr.(*Alloc)
			if !ok {
				continue
			}
			loop, copyInfo := checkArrayPromotion(alloc, blockToLoop)
			if loop != nil {
				candidates = append(candidates, candidate{alloc, loop, copyInfo})
			}
		}
	}

	for _, c := range candidates {
		doPromoteArray(f, c.alloc, c.loop, c.copyInfo)
	}
}

// checkArrayPromotion checks whether alloc is eligible for promotion.
// Returns (loop, nil) for the write-only pattern and (loop, copyInfo) for the
// copy+read pattern. Returns (nil, nil) if ineligible.
func checkArrayPromotion(alloc *Alloc, blockToLoop map[*BasicBlock]*SPMDLoopInfo) (*SPMDLoopInfo, *arrayCopyInfo) {
	// Check 1: type is pointer to fixed-size array.
	ptrType, ok := alloc.Type().Underlying().(*types.Pointer)
	if !ok {
		return nil, nil
	}
	arrayType, ok := ptrType.Elem().Underlying().(*types.Array)
	if !ok {
		return nil, nil
	}

	// Check 2: element type is a plain numeric type (we know its size).
	elemType := arrayType.Elem()
	elemSize := spmdPromoteElemSize(elemType)
	if elemSize == 0 {
		return nil, nil
	}
	arrayLen := int(arrayType.Len())

	// Check 3: fits in SIMD register (default 16 bytes for 128-bit registers).
	regBits := alloc.Parent().Prog.SIMDRegisterBits
	if regBits == 0 {
		regBits = 128
	}
	regBytes := regBits / 8
	if arrayLen*elemSize > regBytes {
		return nil, nil
	}

	// Check 4: categorize referrers into IndexAddr, Slice, and other.
	// Any non-IndexAddr, non-Slice referrer means the alloc escapes.
	refs := alloc.Referrers()
	if refs == nil || len(*refs) == 0 {
		return nil, nil
	}

	var loop *SPMDLoopInfo
	var sliceRef *Slice // Slice instruction for copy+read pattern
	var arrayLoadRef *UnOp // UnOp{MUL} whole-array load (rangeindex pattern)

	for _, ref := range *refs {
		switch r := ref.(type) {
		case *IndexAddr:
			// Handled below.
			_ = r
		case *Slice:
			// Allow at most one Slice referrer (for input[:] in copy pattern).
			if sliceRef != nil {
				return nil, nil // multiple slice ops, too complex
			}
			sliceRef = r
		case *UnOp:
			// Allow one whole-array load (rangeindex over array: t3 = *alloc).
			// This is the array-value pattern where range over [N]T loads the
			// entire array first, then indexes into the loaded value.
			if r.Op != token.MUL {
				return nil, nil // non-load UnOp — ineligible
			}
			if arrayLoadRef != nil {
				return nil, nil // multiple array loads — too complex
			}
			arrayLoadRef = r
		case *DebugRef:
			// ok — debug references don't affect semantics.
		default:
			// Any other referrer (Call directly on alloc, MakeInterface, etc.)
			// means the alloc escapes.
			return nil, nil
		}
	}

	// Check 5: if there is a Slice referrer, verify it feeds a builtin copy call.
	// This detects: t1 = slice alloc[:]; t2 = copy(t1, src)
	var copyInfo *arrayCopyInfo
	if sliceRef != nil {
		ci := checkCopyPattern(sliceRef)
		if ci == nil {
			return nil, nil // Slice not used in copy pattern — ineligible
		}
		copyInfo = ci
	}

	// Check 5b: if there is a whole-array load referrer (rangeindex pattern),
	// require a copy pattern and validate that the loaded value is used only
	// via Index instructions keyed by IterPhi in the SPMD loop body.
	if arrayLoadRef != nil {
		if copyInfo == nil {
			// Array-value reads without external init — ineligible.
			return nil, nil
		}
		copyInfo.arrayLoad = arrayLoadRef

		// Validate all referrers of the loaded array value.
		alRefs := arrayLoadRef.Referrers()
		if alRefs == nil || len(*alRefs) == 0 {
			return nil, nil
		}
		for _, alRef := range *alRefs {
			switch r := alRef.(type) {
			case *Index:
				// t8 = t3[t6]: index into the loaded array value.
				// Validate below after we know the loop (IterPhi).
				copyInfo.indexInstrs = append(copyInfo.indexInstrs, r)
			case *DebugRef:
				// ok — debug references don't affect semantics.
			default:
				// Any other use means the loaded array escapes.
				return nil, nil
			}
		}
	}

	// Now validate all IndexAddr referrers for their access pattern (pattern 1).
	for _, ref := range *refs {
		ia, ok := ref.(*IndexAddr)
		if !ok {
			continue // Slice, UnOp already handled above
		}

		// Check 6: IndexAddr must be inside an SPMD loop body.
		l, ok := blockToLoop[ia.Block()]
		if !ok {
			return nil, nil
		}
		if loop == nil {
			loop = l
		} else if loop != l {
			// References to different loops — too complex.
			return nil, nil
		}

		// Check 7: index must be the IterPhi (possibly via ChangeType).
		// After lift, the iter alloc is promoted to a Phi, and the loop
		// variable `i` is a ChangeType wrapping that Phi. Peel any ChangeType
		// wrappers to reach the underlying Phi.
		if unwrapChangeType(ia.Index) != loop.IterPhi {
			return nil, nil
		}

		// Check 8: validate IndexAddr referrers based on pattern.
		iaRefs := ia.Referrers()
		if iaRefs == nil || len(*iaRefs) == 0 {
			return nil, nil
		}
		for _, iaRef := range *iaRefs {
			switch r := iaRef.(type) {
			case *Store:
				// Write-only pattern: store through the IndexAddr.
				if copyInfo != nil {
					// Can't mix stores and copy+read pattern.
					return nil, nil
				}
				if r.Addr != ia {
					return nil, nil
				}
				if _, ok := blockToLoop[r.Block()]; !ok {
					return nil, nil
				}
			case *UnOp:
				// Load (pointer dereference): read-through the IndexAddr.
				if r.Op != token.MUL {
					return nil, nil // non-load UnOp — ineligible
				}
				if copyInfo == nil {
					// Reads without external init — ineligible (no source data).
					return nil, nil
				}
				if _, ok := blockToLoop[r.Block()]; !ok {
					return nil, nil
				}
				copyInfo.loads = append(copyInfo.loads, r)
			case *DebugRef:
				// ok — debug references don't affect semantics.
			default:
				return nil, nil
			}
		}
	}

	// Check 8b: validate Index instructions for the array-value pattern (pattern 2).
	// These must be in the SPMD loop body and indexed by IterPhi or IncrBinOp.
	// For rangeindex, go/ssa emits: t5=phi(start,-1), t6=t5+1, body uses t6.
	// So the index in the body is IncrBinOp (t6), not IterPhi (t5) directly.
	var indexInstrs []*Index
	if copyInfo != nil {
		indexInstrs = copyInfo.indexInstrs
	}
	for _, idx := range indexInstrs {
		// idx.X is the loaded array value (arrayLoadRef result).
		// idx.Index is the loop iteration variable.

		// Find or validate the loop from the block containing the Index.
		l, ok := blockToLoop[idx.Block()]
		if !ok {
			return nil, nil
		}
		if loop == nil {
			loop = l
		} else if loop != l {
			return nil, nil
		}

		// Index must be keyed by IterPhi or IncrBinOp (possibly via ChangeType).
		// For rangeindex: IterPhi starts at -1, IncrBinOp = IterPhi+1 is used in body.
		unwrapped := unwrapChangeType(idx.Index)
		if unwrapped != loop.IterPhi && (loop.IncrBinOp == nil || unwrapped != Value(loop.IncrBinOp)) {
			return nil, nil
		}
	}

	if loop == nil {
		return nil, nil
	}

	// Check 9: verify the loop iterates exactly arrayLen times.
	// For rangeint, BoundValue is a constant N (e.g., range 16); check it
	// equals arrayLen.
	// For rangeindex over an array, the iteration count equals the array
	// length by construction (range over [N]T iterates exactly N times), so
	// we verify via the loop's LaneCount matching arrayLen instead. We do NOT
	// check BoundValue because for rangeindex it is the array value itself
	// (type [N]T), not a numeric constant.
	if copyInfo != nil && copyInfo.arrayLoad != nil {
		// Array-value pattern: bound is array length by construction.
		// LaneCount >= arrayLen: the SPMD tail mask handles inactive lanes
		// beyond arrayLen. SPMDVectorFromMemory takes Lanes=arrayLen and
		// uses the source length for safe partial loads.
		if loop.LaneCount < arrayLen {
			return nil, nil
		}
	} else {
		// IndexAddr pattern (rangeint): BoundValue must be a constant equal to arrayLen.
		if loop.BoundValue == nil {
			return nil, nil
		}
		bv, ok := loop.BoundValue.(*Const)
		if !ok {
			return nil, nil
		}
		boundInt, ok := constant.Int64Val(bv.Value)
		if !ok {
			return nil, nil
		}
		if int(boundInt) != arrayLen {
			return nil, nil
		}
	}

	return loop, copyInfo
}

// checkCopyPattern verifies that sliceInstr is used as the first argument of a
// builtin copy call and returns the arrayCopyInfo describing the pattern, or nil
// if the pattern does not match. The pattern is:
//
//	t1 = slice alloc[:]      // sliceInstr
//	t2 = copy(t1, src)       // builtin copy call
func checkCopyPattern(sliceInstr *Slice) *arrayCopyInfo {
	sliceRefs := sliceInstr.Referrers()
	if sliceRefs == nil || len(*sliceRefs) == 0 {
		return nil
	}

	// The slice must have exactly one non-DebugRef referrer: the copy call.
	var copyCall *Call
	for _, ref := range *sliceRefs {
		switch r := ref.(type) {
		case *Call:
			if copyCall != nil {
				return nil // more than one call — too complex
			}
			copyCall = r
		case *DebugRef:
			// ok — ignore debug refs.
		default:
			// Slice used somewhere other than a copy call — ineligible.
			return nil
		}
	}
	if copyCall == nil {
		return nil
	}

	// Verify the call is a builtin copy with the slice as first arg.
	cc := &copyCall.Call
	builtin, ok := cc.Value.(*Builtin)
	if !ok || builtin.Name() != "copy" {
		return nil
	}
	if len(cc.Args) < 2 {
		return nil
	}
	if cc.Args[0] != Value(sliceInstr) {
		return nil
	}

	return &arrayCopyInfo{
		sliceInstr: sliceInstr,
		copyCall:   copyCall,
		srcArg:     cc.Args[1],
	}
}

// doPromoteArray removes the alloc and associated instructions.
//
// Write-only pattern (copyInfo == nil): removes the alloc plus all
// Store/IndexAddr instructions that write through it. The stored values are
// left as dead code.
//
// Copy+read pattern (copyInfo != nil): emits SPMDVectorFromMemory to load
// external data directly into a varying value, replaces all UnOp{MUL} load
// results with that value, and removes the copy call, Slice, IndexAddrs, and
// the alloc.
//
// This is safe when all accesses are through IterPhi-indexed IndexAddrs:
// each lane accesses its own position, so no inter-lane aliasing exists.
func doPromoteArray(f *Function, alloc *Alloc, loop *SPMDLoopInfo, copyInfo *arrayCopyInfo) {
	if copyInfo != nil {
		doPromoteCopyRead(f, alloc, loop, copyInfo)
		return
	}

	// Write-only path: unchanged from Chunk 2.
	refs := alloc.Referrers()
	if refs == nil {
		return
	}

	affectedBlocks := make(map[*BasicBlock]bool)

	// Remove DebugRef instructions that reference the alloc directly.
	for _, ref := range *refs {
		if dr, ok := ref.(*DebugRef); ok {
			b := dr.Block()
			spmdNilInstr(b, dr)
			if refs := dr.X.Referrers(); refs != nil {
				*refs = removeInstr(*refs, dr)
			}
			dr.block = nil
			affectedBlocks[b] = true
		}
	}

	// Process each IndexAddr and its Store referrers.
	for _, ref := range *refs {
		ia, ok := ref.(*IndexAddr)
		if !ok {
			continue
		}

		// Remove Store and DebugRef instructions that reference this IndexAddr.
		iaRefs := ia.Referrers()
		if iaRefs != nil {
			for _, iaRef := range *iaRefs {
				switch r := iaRef.(type) {
				case *Store:
					b := r.Block()
					spmdNilInstr(b, r)
					if refs := r.Addr.Referrers(); refs != nil {
						*refs = removeInstr(*refs, r)
					}
					if refs := r.Val.Referrers(); refs != nil {
						*refs = removeInstr(*refs, r)
					}
					r.block = nil
					affectedBlocks[b] = true
				case *DebugRef:
					b := r.Block()
					spmdNilInstr(b, r)
					if refs := r.X.Referrers(); refs != nil {
						*refs = removeInstr(*refs, r)
					}
					r.block = nil
					affectedBlocks[b] = true
				}
			}
		}

		// Remove the IndexAddr itself.
		b := ia.Block()
		spmdNilInstr(b, ia)
		if refs := ia.X.Referrers(); refs != nil {
			*refs = removeInstr(*refs, ia)
		}
		if refs := ia.Index.Referrers(); refs != nil {
			*refs = removeInstr(*refs, ia)
		}
		ia.block = nil
		affectedBlocks[b] = true
	}

	// Remove the alloc itself.
	{
		b := alloc.Block()
		spmdNilInstr(b, alloc)
		alloc.block = nil
		affectedBlocks[b] = true
	}

	// Compact nil slots from all affected blocks.
	for b := range affectedBlocks {
		spmdCompactInstrs(b)
	}
}

// doPromoteCopyRead handles the copy+read pattern:
//
//	var input [N]T
//	copy(input[:], src)    // Slice + copy builtin
//	go for i := range N {
//	    _ = input[i] ...   // IndexAddr + UnOp{MUL} load
//	}
//
// Transforms to:
//
//	%vec = SPMDVectorFromMemory src len(src)   // Varying[T]
//	go for i := range N {
//	    _ = %vec ...       // direct varying value use
//	}
func doPromoteCopyRead(f *Function, alloc *Alloc, loop *SPMDLoopInfo, ci *arrayCopyInfo) {
	ptrType := alloc.Type().Underlying().(*types.Pointer)
	arrayType := ptrType.Elem().Underlying().(*types.Array)
	elemType := arrayType.Elem()
	arrayLen := int(arrayType.Len())

	affectedBlocks := make(map[*BasicBlock]bool)

	// The copy call and slice instruction are in the entry block (block before
	// the loop). Insert the len call and SPMDVectorFromMemory right before the
	// copy call instruction.
	copyBlock := ci.copyCall.Block()

	// Step 1: emit len(src) before the copy call.
	// Use a Call to the builtin len specialised for the source argument's type.
	lenCall := &Call{}
	lenCall.Call.Value = makeLen(ci.srcArg.Type())
	lenCall.Call.Args = []Value{ci.srcArg}
	lenCall.setType(tInt)
	spmdInsertBeforeInstr(copyBlock, ci.copyCall, lenCall)
	spmdAddReferrer(ci.srcArg, lenCall)
	affectedBlocks[copyBlock] = true

	// Step 2: emit SPMDVectorFromMemory before the copy call (after lenCall).
	vec := &SPMDVectorFromMemory{
		Ptr:      ci.srcArg,
		Len:      lenCall,
		ElemType: elemType,
		Lanes:    arrayLen,
	}
	vec.pos = ci.copyCall.Pos()
	vec.setType(vec.Type())
	spmdInsertBeforeInstr(copyBlock, ci.copyCall, vec)
	spmdAddReferrer(ci.srcArg, vec)
	spmdAddReferrer(lenCall, vec)
	affectedBlocks[copyBlock] = true

	if ci.arrayLoad != nil {
		// Pattern 2 (array-value / rangeindex): replace Index instruction results
		// with the varying vector, then remove the whole-array load and Index instrs.

		// Step 3b: replace all Index results with the varying vector.
		for _, idx := range ci.indexInstrs {
			replaceAll(idx, vec)
			b := idx.Block()
			spmdNilInstr(b, idx)
			// Clean up operand referrers for the Index.
			if refs := idx.X.Referrers(); refs != nil {
				*refs = removeInstr(*refs, idx)
			}
			if refs := idx.Index.Referrers(); refs != nil {
				*refs = removeInstr(*refs, idx)
			}
			idx.block = nil
			affectedBlocks[b] = true
		}

		// Step 4b: remove DebugRef instructions on the array load result,
		// then remove the whole-array load itself.
		if alRefs := ci.arrayLoad.Referrers(); alRefs != nil {
			for _, alRef := range *alRefs {
				if dr, ok := alRef.(*DebugRef); ok {
					b := dr.Block()
					spmdNilInstr(b, dr)
					if drRefs := dr.X.Referrers(); drRefs != nil {
						*drRefs = removeInstr(*drRefs, dr)
					}
					dr.block = nil
					affectedBlocks[b] = true
				}
			}
		}
		{
			b := ci.arrayLoad.Block()
			spmdNilInstr(b, ci.arrayLoad)
			if refs := ci.arrayLoad.X.Referrers(); refs != nil {
				*refs = removeInstr(*refs, ci.arrayLoad)
			}
			ci.arrayLoad.block = nil
			affectedBlocks[b] = true
		}
	} else {
		// Pattern 1 (IndexAddr / rangeint): replace UnOp{MUL} load results with
		// the varying vector, then remove IndexAddr instructions.

		// Step 3: replace all UnOp{MUL} load results with the varying vector.
		// replaceAll rewires all users of each load to use vec instead, and sets
		// each load's referrer list to nil (marking it unused).
		for _, load := range ci.loads {
			replaceAll(load, vec)
			b := load.Block()
			spmdNilInstr(b, load)
			// Clean up the load's operand referrer (the IndexAddr).
			if refs := load.X.Referrers(); refs != nil {
				*refs = removeInstr(*refs, load)
			}
			load.block = nil
			affectedBlocks[b] = true
		}

		// Step 4: remove IndexAddr instructions (now unreferenced).
		// Also clean up any DebugRef instructions that reference the IndexAddr
		// result (e.g., for debug info on &arr[i] expressions).
		for _, ref := range *alloc.Referrers() {
			ia, ok := ref.(*IndexAddr)
			if !ok {
				continue
			}
			if iaRefs := ia.Referrers(); iaRefs != nil {
				for _, iaRef := range *iaRefs {
					if dr, ok := iaRef.(*DebugRef); ok {
						b := dr.Block()
						spmdNilInstr(b, dr)
						if drRefs := dr.X.Referrers(); drRefs != nil {
							*drRefs = removeInstr(*drRefs, dr)
						}
						dr.block = nil
						affectedBlocks[b] = true
					}
				}
			}
			b := ia.Block()
			spmdNilInstr(b, ia)
			if refs := ia.X.Referrers(); refs != nil {
				*refs = removeInstr(*refs, ia)
			}
			if refs := ia.Index.Referrers(); refs != nil {
				*refs = removeInstr(*refs, ia)
			}
			ia.block = nil
			affectedBlocks[b] = true
		}
	}

	// Step 5: remove the copy call.
	{
		b := ci.copyCall.Block()
		spmdNilInstr(b, ci.copyCall)
		// Clean up operand referrers for the copy call.
		var rands []*Value
		for _, rand := range ci.copyCall.Operands(rands) {
			if *rand != nil {
				if refs := (*rand).Referrers(); refs != nil {
					*refs = removeInstr(*refs, ci.copyCall)
				}
			}
		}
		ci.copyCall.block = nil
		affectedBlocks[b] = true
	}

	// Step 6: remove the Slice instruction.
	{
		b := ci.sliceInstr.Block()
		spmdNilInstr(b, ci.sliceInstr)
		if refs := ci.sliceInstr.X.Referrers(); refs != nil {
			*refs = removeInstr(*refs, ci.sliceInstr)
		}
		ci.sliceInstr.block = nil
		affectedBlocks[b] = true
	}

	// Step 7: remove DebugRef instructions that reference the alloc directly.
	if refs := alloc.Referrers(); refs != nil {
		for _, ref := range *refs {
			if dr, ok := ref.(*DebugRef); ok {
				b := dr.Block()
				spmdNilInstr(b, dr)
				if drRefs := dr.X.Referrers(); drRefs != nil {
					*drRefs = removeInstr(*drRefs, dr)
				}
				dr.block = nil
				affectedBlocks[b] = true
			}
		}
	}

	// Step 8: remove the alloc itself.
	{
		b := alloc.Block()
		spmdNilInstr(b, alloc)
		alloc.block = nil
		affectedBlocks[b] = true
	}

	// Compact nil slots from all affected blocks.
	for b := range affectedBlocks {
		spmdCompactInstrs(b)
	}
}

// spmdInsertBeforeInstr inserts newInstr into block b immediately before
// target. Sets newInstr.block = b. Panics if target is not found in b.
func spmdInsertBeforeInstr(b *BasicBlock, target, newInstr Instruction) {
	newInstr.setBlock(b)
	for i, ins := range b.Instrs {
		if ins == target {
			// Grow slice by one and shift tail right.
			b.Instrs = append(b.Instrs, nil)
			copy(b.Instrs[i+1:], b.Instrs[i:])
			b.Instrs[i] = newInstr
			return
		}
	}
	panic("spmdInsertBeforeInstr: target not found in block")
}

// unwrapChangeType peels any ChangeType wrappers from v, returning the
// underlying value. If v is not a ChangeType, v is returned unchanged.
// This handles the pattern where lift promotes an alloc to a Phi and the
// loop variable becomes a ChangeType wrapping that Phi.
func unwrapChangeType(v Value) Value {
	for {
		ct, ok := v.(*ChangeType)
		if !ok {
			return v
		}
		v = ct.X
	}
}

// spmdNilInstr sets the slot for instr to nil in b.Instrs.
// The caller is responsible for calling spmdCompactInstrs afterwards to
// remove nil gaps from the instruction slice.
func spmdNilInstr(b *BasicBlock, instr Instruction) {
	for i, ins := range b.Instrs {
		if ins == instr {
			b.Instrs[i] = nil
			return
		}
	}
}

// spmdPromoteElemSize returns the byte size of a basic numeric type,
// or 0 if the type is unsupported (non-numeric or platform-dependent size).
// Platform-dependent types (int, uint, uintptr) are excluded because their
// size varies between WASM32 and amd64, making lane count computation unsafe.
func spmdPromoteElemSize(t types.Type) int {
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return 0
	}
	switch basic.Kind() {
	case types.Byte, types.Int8: // types.Byte == types.Uint8
		return 1
	case types.Int16, types.Uint16:
		return 2
	case types.Int32, types.Uint32, types.Float32:
		return 4
	case types.Int64, types.Uint64, types.Float64:
		return 8
	default:
		// Platform-dependent sizes (int, uint, uintptr) or unsupported types.
		return 0
	}
}
