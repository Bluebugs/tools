// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/constant"
	"go/types"
)

// promoteSPMDArrays promotes eligible small array allocations inside SPMD
// loops. An Alloc is eligible when:
//
//   - Type is *[N]T where N*sizeof(T) <= 16 (fits in v128)
//   - The loop BoundValue is a constant equal to N (array length == loop bound)
//   - All referrers are IndexAddr with index == unwrapChangeType(IterPhi)
//   - All IndexAddr referrers are Store (as Addr) or DebugRef
//   - All use sites are inside the enclosing SPMD loop body
//
// This pass runs after resolveSPMDLoops (so IterPhi is available) and before
// predicateSPMD (so stores are still regular *Store instructions from the
// builder, not yet converted to SPMDStore by spmdConvertLoopOps). Removing
// write-only array allocations here prevents TinyGo from generating unnecessary
// per-lane GEP + scatter sequences for simple lane-indexed writes.
func promoteSPMDArrays(f *Function) {
	if len(f.SPMDLoops) == 0 {
		return
	}

	// Build a map from block to its enclosing SPMDLoopInfo.
	blockToLoop := make(map[*BasicBlock]*SPMDLoopInfo)
	for _, loop := range f.SPMDLoops {
		if loop.BodyBlock != nil {
			blockToLoop[loop.BodyBlock] = loop
		}
		if loop.LoopBlock != nil {
			blockToLoop[loop.LoopBlock] = loop
		}
	}

	type candidate struct {
		alloc *Alloc
		loop  *SPMDLoopInfo
	}
	var candidates []candidate

	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			alloc, ok := instr.(*Alloc)
			if !ok {
				continue
			}
			if loop := checkArrayPromotion(alloc, blockToLoop); loop != nil {
				candidates = append(candidates, candidate{alloc, loop})
			}
		}
	}

	for _, c := range candidates {
		doPromoteArray(f, c.alloc, c.loop)
	}
}

// checkArrayPromotion returns the enclosing SPMDLoopInfo if alloc is eligible
// for promotion, or nil otherwise.
func checkArrayPromotion(alloc *Alloc, blockToLoop map[*BasicBlock]*SPMDLoopInfo) *SPMDLoopInfo {
	// Check 1: type is pointer to fixed-size array.
	ptrType, ok := alloc.Type().Underlying().(*types.Pointer)
	if !ok {
		return nil
	}
	arrayType, ok := ptrType.Elem().Underlying().(*types.Array)
	if !ok {
		return nil
	}

	// Check 2: element type is a plain numeric type (we know its size).
	elemType := arrayType.Elem()
	elemSize := spmdPromoteElemSize(elemType)
	if elemSize == 0 {
		return nil
	}
	arrayLen := int(arrayType.Len())

	// Check 3: fits in v128 (16 bytes).
	if arrayLen*elemSize > 16 {
		return nil
	}

	// Check 4: all referrers must be IndexAddr instructions (no escapes).
	refs := alloc.Referrers()
	if refs == nil || len(*refs) == 0 {
		return nil
	}
	var loop *SPMDLoopInfo
	for _, ref := range *refs {
		ia, ok := ref.(*IndexAddr)
		if !ok {
			// Any non-IndexAddr referrer (call, MakeInterface, etc.) means escape.
			return nil
		}

		// Check 5: IndexAddr is inside an SPMD loop body.
		l, ok := blockToLoop[ia.Block()]
		if !ok {
			return nil
		}
		if loop == nil {
			loop = l
		} else if loop != l {
			// References to different loops — too complex.
			return nil
		}

		// Check 6: index must be the IterPhi (possibly via ChangeType).
		// After lift, the iter alloc is promoted to a Phi, and the loop
		// variable `i` is a ChangeType wrapping that Phi. Peel any ChangeType
		// wrappers to reach the underlying Phi.
		if unwrapChangeType(ia.Index) != loop.IterPhi {
			return nil
		}

		// Check 7: IndexAddr referrers must be Store (as Addr) or DebugRef.
		// At this point in the pipeline (after lift, before spmdConvertLoopOps),
		// stores are still plain *Store instructions. Only write-through access
		// is allowed; any read access (UnOp{MUL} load) makes the array ineligible
		// because the stored values would need to be tracked across lanes.
		iaRefs := ia.Referrers()
		if iaRefs == nil || len(*iaRefs) == 0 {
			return nil
		}
		for _, iaRef := range *iaRefs {
			switch r := iaRef.(type) {
			case *Store:
				if r.Addr != ia {
					return nil
				}
				if _, ok := blockToLoop[r.Block()]; !ok {
					return nil
				}
			case *DebugRef:
				// ok — debug references don't affect semantics.
			default:
				return nil
			}
		}
	}

	if loop == nil {
		return nil
	}

	// Check 8: loop BoundValue is a constant equal to arrayLen.
	// This ensures the array exactly covers one iteration of the loop
	// (every element is written exactly once, one per lane). Non-constant
	// bounds or mismatched lengths are ineligible.
	if loop.BoundValue == nil {
		return nil
	}
	bv, ok := loop.BoundValue.(*Const)
	if !ok {
		return nil
	}
	boundInt, ok := constant.Int64Val(bv.Value)
	if !ok {
		return nil
	}
	if int(boundInt) != arrayLen {
		return nil
	}

	return loop
}

// doPromoteArray removes the alloc and all Store/IndexAddr instructions
// that write through it. The stored values (representing per-lane data) are
// left as dead code — they will either be removed by subsequent DCE or remain
// harmlessly as unused values.
//
// This is safe when all writes are through IterPhi-indexed IndexAddrs:
// each lane writes to its own position, so no inter-lane aliasing exists.
// The alloc was write-only (no loads), so removing the writes loses no data.
func doPromoteArray(f *Function, alloc *Alloc, loop *SPMDLoopInfo) {
	refs := alloc.Referrers()
	if refs == nil {
		return
	}

	affectedBlocks := make(map[*BasicBlock]bool)

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
