// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// predicateSPMD transforms varying control flow in SPMD functions into
// predicated form with explicit mask-gated operations.
//
// For each varying If (If.IsVarying == true) in an SPMD loop scope:
//   - The If is replaced with a Jump to the then-block.
//   - The then-block falls through to the else-block (or merge).
//   - Phi instructions at the merge block are replaced with SPMDSelect.
//   - UnOp{MUL} (pointer loads) and Store instructions in varying paths
//     are replaced with SPMDLoad and SPMDStore.
//
// This pass runs after resolveSPMDLoops and before numberRegisters in finishBody.

import (
	"go/constant"
	"go/token"
	"go/types"
	"slices"
	"strings"

	spmdpkg "golang.org/x/tools/go/types/spmd"
)

// spmdPhiSnapshot captures the edge values of a Phi before CFG rewiring.
// Used to reconstruct Phi→SPMDSelect when predecessors are removed.
type spmdPhiSnapshot struct {
	phi     *Phi
	edgeVal map[*BasicBlock]Value // predecessor → edge value
}

// spmdDeferredMerge captures a varying-if where both branches merge at a loop
// header. The If is linearized (replaced with Jump, mem ops masked) but the
// loop-header phis are NOT converted to SPMDSelect during predicateVaryingIf
// because the merge block has >2 predecessors. Instead, this info is collected
// and processed after all varying-Ifs in the scope are linearized.
//
// After CFG rewiring (spmdReplaceIfWithJump + spmdRewireThenToElse), the
// thenBlock no longer jumps to the loop header — it jumps to elseBlock instead.
// The loopHeader retains elseBlock as a predecessor but loses thenBlock.
// The phi edge values from thenBlock are snapshotted in phiSnaps before rewiring
// so they can be used to build SPMDSelect instructions later.
type spmdDeferredMerge struct {
	thenBlock  *BasicBlock
	elseBlock  *BasicBlock
	mergeBlock *BasicBlock // the loop header
	thenMask   Value
	elseMask   Value
	lanes      int
	phiSnaps   []spmdPhiSnapshot
}

// hasSPMDParams reports whether any parameter of fn has *types.SPMDType.
// Such functions are "SPMD function bodies" that process per-lane varying values.
func hasSPMDParams(fn *Function) bool {
	sig := fn.Signature
	if sig == nil {
		return false
	}
	params := sig.Params()
	for i := 0; i < params.Len(); i++ {
		if _, ok := params.At(i).Type().(*types.SPMDType); ok {
			return true
		}
	}
	return false
}

// predicateSPMD transforms varying control flow in SPMD functions with go-for
// loops. Called from finishBody after resolveSPMDLoops.
// Handles functions with SPMDLoops (go-for loops): linearizes varying Ifs in
// each loop's scope.
// For SPMD function bodies with no go-for loops, finishBody calls
// predicateSPMDFuncBody directly after the SPMDLoops block.
func predicateSPMD(fn *Function) {
	if len(fn.SPMDLoops) == 0 {
		return
	}

	for _, loop := range fn.SPMDLoops {
		predicateSPMDLoop(fn, loop)
	}
}

// predicateSPMDLoop transforms varying control flow within a single SPMD loop.
// It performs two passes over the loop body blocks:
//
//  1. predicateSPMDScope linearizes varying Ifs (if/else, switch, boolean chains)
//     within the loop scope. For loop-header merge patterns — where both then and
//     else branches jump to the loop header — the varying If is linearized (B→T→E)
//     but phi conversion is deferred because the loop header has >2 predecessors.
//
//  2. spmdConvertDeferredMerges resolves the deferred loop-header phi→SPMDSelect
//     conversions collected during pass 1.
//
// Straight-line mem op conversion (spmdConvertScopedMemOps) and SPMD call masking
// (spmdMaskScopedCallOps) are NOT called here because this function runs BEFORE
// peelSPMDLoops. Converting mem ops before peeling would leave orphaned referrers
// in the original body block after deleteUnreachableBlocks removes it. Instead,
// spmdConvertLoopOps (called from func.go after peeling and cleanup) handles those
// passes on the live peeled blocks.
func predicateSPMDLoop(fn *Function, loop *SPMDLoopInfo) {
	scopeBlocks := spmdLoopScopeBlocks(loop)
	activeMask := spmdAllOnesMask()

	// Pass 1: linearize varying control flow; collect deferred loop-header merges.
	var deferred []*spmdDeferredMerge
	predicateSPMDScope(fn, scopeBlocks, loop.LaneCount, loop.LoopBlock, loop.BodyBlock, activeMask, &deferred)

	// Pass 2: convert deferred loop-header phi→SPMDSelect.
	spmdConvertDeferredMerges(fn, deferred)
}

// spmdConvertLoopOps converts straight-line mem ops and masks SPMD function calls
// in all live SPMD loop scope blocks. Called after peelSPMDLoops and
// deleteUnreachableBlocks so that peeled blocks (spmd.main.body, spmd.tail.body)
// are live, and the original (pre-peeling) body blocks that were converted and
// then made unreachable by peeling have been removed.
//
// For each SPMDLoopInfo, the scope blocks are recomputed from the current live
// function blocks (post-peeling) by BFS from the loop/body blocks. Because
// deleteUnreachableBlocks has already run, any unreachable cloned blocks are gone.
// The BFS correctly enumerates only live scope blocks.
func spmdConvertLoopOps(fn *Function) {
	if len(fn.SPMDLoops) == 0 {
		return
	}
	allOnesMask := spmdAllOnesMask()
	for _, loop := range fn.SPMDLoops {
		// Recompute scope blocks post-peeling: BFS over live blocks.
		// spmdLoopScopeBlocks uses loop.BodyBlock and loop.LoopBlock as roots;
		// after peeling, these may be the original blocks (now unreachable and
		// deleted) OR the peeled mainBody (if the loop was MergedBodyLoop and got
		// peeled). In either case, the BFS stops at loop.DoneBlock so only
		// in-loop blocks are visited.
		//
		// For non-peeled loops (rangeindex, multi-block, unsupported), the
		// original scope blocks are still live and the BFS is correct as-is.
		//
		// For peeled rangeint loops, loop.BodyBlock may now be unreachable (the
		// original merged body+loop block was cloned into mainBody and tailBody).
		// We detect this and fall back to scanning all live blocks that are
		// inside an SPMD loop scope by checking for the "spmd." comment prefix.
		scopeBlocks := spmdLoopScopeBlocks(loop)
		// Intersect with live blocks (fn.Blocks after deleteUnreachableBlocks).
		liveSet := make(map[*BasicBlock]bool, len(fn.Blocks))
		for _, b := range fn.Blocks {
			liveSet[b] = true
		}
		liveScopeBlocks := make(map[*BasicBlock]bool)
		for b := range scopeBlocks {
			if liveSet[b] {
				liveScopeBlocks[b] = true
			}
		}
		// If BFS yielded no live scope blocks (peeled loop with original body
		// removed), fall back to BFS from EntryBlock through peeled blocks
		// (spmd.main.*, spmd.tail.*) bounded by DoneBlock. This correctly
		// scopes the search to blocks belonging to THIS loop, avoiding
		// cross-contamination in multi-loop functions.
		if len(liveScopeBlocks) == 0 && loop.EntryBlock != nil {
			queue := []*BasicBlock{loop.EntryBlock}
			visited := map[*BasicBlock]bool{loop.EntryBlock: true}
			for len(queue) > 0 {
				b := queue[0]
				queue = queue[1:]
				if b == loop.DoneBlock {
					continue
				}
				if strings.HasPrefix(b.Comment, "spmd.main.") ||
					strings.HasPrefix(b.Comment, "spmd.tail.") {
					liveScopeBlocks[b] = true
				}
				for _, succ := range b.Succs {
					if !visited[succ] {
						visited[succ] = true
						queue = append(queue, succ)
					}
				}
			}
		}
		if len(liveScopeBlocks) == 0 {
			continue
		}

		// Create tail mask virtual parameter for this loop if peeled.
		if loop.IsPeeled && loop.TailMask == nil {
			loop.TailMask = &Parameter{name: "spmd.tail.mask", typ: spmdpkg.NewVaryingMask(), parent: fn}
		}

		// Separate main vs tail blocks so each phase gets the appropriate mask.
		tailBlocks := spmdTailScopeBlocks(loop, liveScopeBlocks)
		mainBlocks := make(map[*BasicBlock]bool)
		for b := range liveScopeBlocks {
			if !tailBlocks[b] {
				mainBlocks[b] = true
			}
		}

		// Convert main blocks with all-ones mask (every lane is active).
		spmdConvertScopedMemOps(fn, mainBlocks, allOnesMask, loop.LaneCount)
		spmdMaskScopedCallOps(fn, mainBlocks, allOnesMask)
		spmdMaskScopedIndexOps(fn, mainBlocks, allOnesMask)

		// Convert tail blocks with the tail mask (partial last iteration).
		tailMask := Value(allOnesMask)
		if loop.TailMask != nil {
			tailMask = loop.TailMask
		}
		spmdConvertScopedMemOps(fn, tailBlocks, tailMask, loop.LaneCount)
		spmdMaskScopedCallOps(fn, tailBlocks, tailMask)
		spmdMaskScopedIndexOps(fn, tailBlocks, tailMask)
	}
}

// predicateSPMDScope linearizes varying control flow in scopeBlocks.
// spmdLoopBlock and spmdBodyBlock identify the SPMD loop header blocks (used
// by switch predication to avoid replacing loop-carried phis). Pass nil for
// both when predicating a function body (no enclosing SPMD loop).
//
// deferred, when non-nil, collects loop-header merge patterns encountered during
// the varying-If pass. These are handled by the caller after this function returns
// via spmdConvertDeferredMerges. Pass nil when predicating a function body (no
// deferred merges needed because the trampoline approach is used instead).
func predicateSPMDScope(fn *Function, scopeBlocks map[*BasicBlock]bool, lanes int, spmdLoopBlock, spmdBodyBlock *BasicBlock, activeMask Value, deferred *[]*spmdDeferredMerge) {
	// Build a set of If instructions to exclude from the generic If predication
	// pass. Both boolean-chain and switch-chain Ifs are excluded so the generic
	// loop below does not linearize them independently. Switch-chain Ifs are
	// handled by predicateVaryingSwitch below. Boolean chains are handled by
	// predicateBooleanChain below.
	excludedIfs := spmdBuildExcludedIfs(fn)

	// Linearize each varying boolean chain whose first block is in scope.
	// Process boolean chains before the generic If pass so that after
	// collapsing the chain blocks into a linear sequence, no remaining
	// varying Ifs from those blocks survive into the generic loop below.
	for _, chain := range fn.SPMDBooleanChains {
		if !chain.IsVarying {
			continue
		}
		if len(chain.Blocks) == 0 {
			continue
		}
		if !scopeBlocks[chain.Blocks[0]] {
			continue
		}
		// Guard: skip if the first block's If has already been linearized
		// (block() == nil after spmdReplaceIfWithJump).
		if len(chain.Blocks[0].Instrs) == 0 {
			continue
		}
		if _, ok := chain.Blocks[0].Instrs[len(chain.Blocks[0].Instrs)-1].(*If); !ok {
			continue
		}
		predicateBooleanChain(fn, lanes, chain, activeMask, deferred)
	}

	// Find all varying If instructions in scope and linearize them.
	// We iterate fn.Blocks in program order, which is sufficient for
	// non-nested varying Ifs. For nested mask threading (Phase 8),
	// dominator-tree order will be needed.
	//
	// In SPMD function body context (spmdLoopBlock == nil), loop-header merges
	// are handled by inserting a trampoline merge block (allowLoopHeaderMerge=true).
	// In go-for loop context (spmdLoopBlock != nil), trampolines are not inserted
	// because they could break SSA-level loop peeling. Instead, loop-header merge
	// patterns are linearized and deferred into the provided deferred slice for
	// phi→SPMDSelect conversion after all Ifs in the scope are processed.
	allowLoopHeaderMerge := spmdLoopBlock == nil
	for _, block := range fn.Blocks {
		if !scopeBlocks[block] {
			continue
		}
		vif, ok := block.Instrs[len(block.Instrs)-1].(*If)
		if !ok || !vif.IsVarying {
			continue
		}
		if excludedIfs[vif] {
			continue // skip boolean-chain and switch-chain Ifs
		}
		// linearize this varying If.
		predicateVaryingIf(fn, lanes, block, vif, allowLoopHeaderMerge, activeMask, deferred)
	}

	// Linearize each varying switch chain whose DoneBlock is in scope.
	for _, chain := range fn.SPMDSwitchChains {
		if chain.DoneBlock == nil {
			continue
		}
		// Check that at least one case block is in scope.
		if len(chain.Cases) == 0 {
			continue
		}
		firstBlock := chain.Cases[0].Block()
		if firstBlock == nil {
			continue // already linearized by a prior pass
		}
		if !scopeBlocks[firstBlock] {
			continue
		}
		predicateVaryingSwitch(fn, lanes, chain, spmdLoopBlock, spmdBodyBlock, activeMask)
	}
}

// ----------------------------------------------------------------------------
// SPMD function body predication (Phase 8)
//
// Functions with *types.SPMDType parameters ("SPMD bodies") may contain regular
// for-range loops. Inside such loops, a varying if with a break is a "varying
// break" — the break must accumulate into a break mask rather than immediately
// jumping to the loop done block.

// spmdRegularForLoop represents a regular for-range loop detected inside an
// SPMD function body. Unlike SPMDLoopInfo (which covers go-for SPMD loops),
// this represents plain Go for loops.
type spmdRegularForLoop struct {
	loopBlock *BasicBlock // rangeint.loop (or body when merged)
	bodyBlock *BasicBlock // rangeint.body
	doneBlock *BasicBlock // rangeint.done (loop exit)
	iterPhi   *Phi        // rangeint.iter phi (for back-edge identification)
}

// spmdVaryingBreak represents a varying break inside a regular for-loop.
// The varying If's then-branch jumps directly to the loop's done block.
type spmdVaryingBreak struct {
	ifBlock   *BasicBlock // block containing the varying If
	thenBlock *BasicBlock // then-branch that jumps to done (the break block)
	elseBlock *BasicBlock // else-branch (continuation; typically jumps back to loop)
	vif       *If         // the varying If instruction
}

// predicateSPMDFuncBody transforms varying control flow in SPMD function bodies
// (functions with SPMDType parameters but no go-for loops).
// Step 1 linearizes all varying if/else/switch/boolean-chain control flow in
// the entire function body via predicateSPMDScope.
// Step 2 handles varying breaks inside regular for-range loops by transforming
// them into break mask accumulation via predicateVaryingBreaks.
func predicateSPMDFuncBody(fn *Function) {
	lanes := spmdFuncBodyLaneCount(fn)
	if lanes == 0 {
		return
	}

	// Determine the active mask for this function body.
	var activeMask Value
	if fn.SPMDMask != nil {
		activeMask = fn.SPMDMask
	} else {
		activeMask = spmdAllOnesMask()
	}

	// Step 1: General varying control flow linearization.
	// Compute scope = ALL function blocks.
	scope := make(map[*BasicBlock]bool, len(fn.Blocks))
	for _, b := range fn.Blocks {
		scope[b] = true
	}
	predicateSPMDScope(fn, scope, lanes, nil, nil, activeMask, nil)

	// Step 2: Varying breaks in regular for-loops.
	// Note: Step 1 does not consume break-case varying Ifs because
	// findMergeBlock returns nil for the break pattern (then-block jumps
	// to doneBlock, else-block jumps to loopBlock — they diverge). Those
	// Ifs survive Step 1 and are handled here via break mask accumulation.
	forLoops := spmdFindRegularForLoops(fn)
	for _, fl := range forLoops {
		vbreaks := spmdFindVaryingBreaks(fn, fl)
		if len(vbreaks) == 0 {
			continue
		}
		predicateVaryingBreaks(fn, fl, vbreaks, lanes, activeMask)
	}

	// Step 2.5: Convert remaining straight-line loads/stores to SPMDLoad/SPMDStore.
	// After Step 1 (which converted loads/stores inside varying branches),
	// there may still be plain UnOp{MUL}/Store instructions in straight-line
	// code. Convert them so TinyGo has a uniform instruction set to handle.
	spmdConvertAllMemOps(fn, activeMask, lanes)

	// Step 2.6: Set mask on IndexAddr/Index instructions with varying indices.
	// This enables TinyGo to clamp inactive-lane indices to 0 at compile time.
	spmdMaskAllIndexOps(fn, activeMask)

	// Step 3: Set mask on remaining SPMD function calls that weren't inside
	// a varying if/else/switch (which already got narrowed masks).
	spmdMaskCallOps(fn, activeMask)
}

// spmdFuncBodyLaneCount returns the lane count for an SPMD function body by
// inspecting its first SPMDType parameter. Returns 0 if none is found.
func spmdFuncBodyLaneCount(fn *Function) int {
	if fn.Signature == nil {
		return 0
	}
	params := fn.Signature.Params()
	for i := 0; i < params.Len(); i++ {
		stype, ok := params.At(i).Type().(*types.SPMDType)
		if !ok {
			continue
		}
		// Compute lane count from element type: 128 / element-size-in-bits.
		// Use the same formula as the type checker (128-bit SIMD register).
		elem := stype.Elem()
		bits := spmdElemBits(elem)
		if bits > 0 {
			return 128 / bits
		}
	}
	return 0
}

// spmdElemBits returns the bit width of a basic type suitable for SIMD lane count
// computation. Returns 0 for unknown or non-basic types.
func spmdElemBits(t types.Type) int {
	basic, ok := t.(*types.Basic)
	if !ok {
		return 0
	}
	switch basic.Kind() {
	case types.Bool, types.Int8, types.Uint8:
		return 8
	case types.Int16, types.Uint16:
		return 16
	case types.Int32, types.Uint32, types.Float32:
		return 32
	case types.Int64, types.Uint64, types.Float64:
		return 64
	case types.Int, types.Uint, types.Uintptr:
		// Platform-dependent; default to 64-bit for lane count computation.
		return 64
	}
	return 0
}

// spmdFindRegularForLoops finds regular for-range loops in fn that match the
// rangeint SSA pattern: a block with comment "rangeint.body" containing an
// iter phi with comment "rangeint.iter", paired with a "rangeint.done" block.
//
// These are plain Go for loops (not SPMD go-for loops) that appear inside
// SPMD function bodies.
func spmdFindRegularForLoops(fn *Function) []*spmdRegularForLoop {
	var loops []*spmdRegularForLoop

	// Index blocks already covered by SPMDLoopInfo to avoid double-processing.
	spmdBodyBlocks := make(map[*BasicBlock]bool)
	for _, info := range fn.SPMDLoops {
		spmdBodyBlocks[info.BodyBlock] = true
		spmdBodyBlocks[info.LoopBlock] = true
	}

	// Build a map from block comment prefix to block for done-block lookup.
	// We match by comment prefix to handle suffixes like ":3" after optimization.
	doneByPrefix := make(map[string]*BasicBlock)
	for _, block := range fn.Blocks {
		if strings.Contains(block.Comment, "rangeint.done") {
			doneByPrefix[block.Comment] = block
		}
	}

	for _, block := range fn.Blocks {
		if spmdBodyBlocks[block] {
			continue // skip SPMD loop blocks
		}
		// Look for rangeint.body blocks (comment contains "rangeint.body").
		if !strings.Contains(block.Comment, "rangeint.body") {
			continue
		}
		// Find the iter phi in this block.
		var iterPhi *Phi
		for _, instr := range block.Instrs {
			phi, ok := instr.(*Phi)
			if !ok {
				break
			}
			if strings.Contains(phi.Comment, "rangeint.iter") {
				iterPhi = phi
				break
			}
		}
		if iterPhi == nil {
			continue
		}

		// Identify loop block (predecessor with "rangeint.loop" comment or same block).
		// For merged body+loop (most rangeint cases), the body block IS the loop block.
		loopBlock := block
		for _, pred := range block.Preds {
			if strings.Contains(pred.Comment, "rangeint.loop") {
				loopBlock = pred
				break
			}
		}

		// Find the done block: look for the block with "rangeint.done" comment
		// that is reachable from this loop (shares the same function scope).
		// For functions with a single loop, there is exactly one such block.
		// For multiple loops, we match by BFS from the loop body.
		doneBlock := spmdFindLoopDoneBlock(fn, block, loopBlock, doneByPrefix)
		if doneBlock == nil {
			continue
		}

		loops = append(loops, &spmdRegularForLoop{
			loopBlock: loopBlock,
			bodyBlock: block,
			doneBlock: doneBlock,
			iterPhi:   iterPhi,
		})
	}
	return loops
}

// spmdFindLoopDoneBlock finds the done block for a regular for-loop.
// It first tries to find a "rangeint.done" block reachable from bodyBlock
// without re-entering the loop (stops at loopBlock to avoid infinite loops).
// Falls back to searching the doneByPrefix map if BFS finds only one candidate.
func spmdFindLoopDoneBlock(fn *Function, bodyBlock, loopBlock *BasicBlock, doneByPrefix map[string]*BasicBlock) *BasicBlock {
	// BFS from bodyBlock, stopping at loopBlock (back-edge).
	// Collect all "rangeint.done" blocks encountered.
	visited := make(map[*BasicBlock]bool)
	queue := []*BasicBlock{bodyBlock}
	visited[bodyBlock] = true
	if loopBlock != bodyBlock {
		visited[loopBlock] = true // don't re-enter loop block
	}
	var candidates []*BasicBlock
	for len(queue) > 0 {
		b := queue[0]
		queue = queue[1:]
		if strings.Contains(b.Comment, "rangeint.done") {
			candidates = append(candidates, b)
			continue // don't explore past the done block
		}
		for _, succ := range b.Succs {
			if visited[succ] {
				continue
			}
			visited[succ] = true
			queue = append(queue, succ)
		}
	}
	if len(candidates) == 1 {
		return candidates[0]
	}
	// Ambiguous or not found: if there is exactly one "rangeint.done" in the
	// entire function, use it (common single-loop case).
	if len(doneByPrefix) == 1 {
		for _, b := range doneByPrefix {
			return b
		}
	}
	return nil
}

// spmdFindVaryingBreaks finds varying break patterns inside a regular for-loop.
// A varying break is: a varying If (IsVarying=true) in the loop body whose
// then-block (Succs[0]) has a single Jump to the loop's done block.
func spmdFindVaryingBreaks(fn *Function, fl *spmdRegularForLoop) []*spmdVaryingBreak {
	// Build the set of blocks in the loop body (between bodyBlock and doneBlock).
	loopBodyBlocks := spmdRegularLoopScopeBlocks(fl)

	var breaks []*spmdVaryingBreak
	for block := range loopBodyBlocks {
		if len(block.Instrs) == 0 {
			continue
		}
		vif, ok := block.Instrs[len(block.Instrs)-1].(*If)
		if !ok || !vif.IsVarying {
			continue
		}
		thenBlock := block.Succs[0]
		elseBlock := block.Succs[1]

		// Check if thenBlock is a break block: single instruction (Jump) to doneBlock.
		if !spmdIsBreakBlock(thenBlock, fl.doneBlock) {
			continue
		}
		breaks = append(breaks, &spmdVaryingBreak{
			ifBlock:   block,
			thenBlock: thenBlock,
			elseBlock: elseBlock,
			vif:       vif,
		})
	}
	return breaks
}

// spmdRegularLoopScopeBlocks returns the set of blocks in a regular for-loop's
// body: blocks reachable from bodyBlock without passing through doneBlock.
func spmdRegularLoopScopeBlocks(fl *spmdRegularForLoop) map[*BasicBlock]bool {
	scope := make(map[*BasicBlock]bool)
	queue := []*BasicBlock{fl.bodyBlock}
	scope[fl.bodyBlock] = true
	if fl.loopBlock != fl.bodyBlock {
		queue = append(queue, fl.loopBlock)
		scope[fl.loopBlock] = true
	}
	for len(queue) > 0 {
		b := queue[0]
		queue = queue[1:]
		for _, succ := range b.Succs {
			if succ == fl.doneBlock || scope[succ] {
				continue
			}
			scope[succ] = true
			queue = append(queue, succ)
		}
	}
	return scope
}

// spmdIsBreakBlock reports whether block ends with a single Jump to doneBlock
// and contains only the Jump instruction (i.e., it is a bare break block with
// no other computation). Also accepts blocks that are the then-branch of a
// varying If and directly jump to doneBlock with at most one instruction.
func spmdIsBreakBlock(block, doneBlock *BasicBlock) bool {
	// The block must have exactly one successor: the done block.
	if len(block.Succs) != 1 || block.Succs[0] != doneBlock {
		return false
	}
	// The block must end with a Jump (not another If or Return).
	if len(block.Instrs) == 0 {
		return false
	}
	_, ok := block.Instrs[len(block.Instrs)-1].(*Jump)
	return ok
}

// predicateVaryingBreaks transforms varying breaks in a regular for-loop into
// break mask accumulation. For each varying break, the then-block (which jumped
// to doneBlock) is linearized: control flows to elseBlock instead, and the
// break lanes are accumulated into a break mask phi at the loop header.
//
// For each result phi at doneBlock that has a break-edge value, an accumulator
// phi is also created at the loop header, carrying the break-selected result
// across iterations. This ensures the accumulated break result dominates the
// done block (since the accumulator phi lives in the loop header, which
// dominates the done block).
//
// Transformed structure:
//
//	loopBlock:
//	    break_mask  = phi [entry: zero_mask, back_edge: new_break_mask]
//	    result_accum = phi [entry: default_val, back_edge: sel_result]  (per result phi)
//	    jump bodyBlock
//
//	bodyBlock / ifBlock:
//	    active_mask = AND_NOT(all_ones, break_mask)
//	    mask_cond   = convert(vif.Cond)
//	    break_lanes = AND(active_mask, mask_cond)
//	    new_break_mask = OR(break_mask, break_lanes)
//	    sel_result = SPMDSelect(break_lanes, break_val, result_accum, Lanes)
//	    jump elseBlock  (linearized: no jump to doneBlock)
//
//	doneBlock:
//	    result = result_accum  (replaced with reference to accumulator phi)
func predicateVaryingBreaks(fn *Function, fl *spmdRegularForLoop, breaks []*spmdVaryingBreak, lanes int, activeMask Value) {
	// Step A: Identify the entry and back-edge predecessors of the loop header.
	loopBlock := fl.loopBlock
	entryPred, _ := spmdLoopHeaderPreds(loopBlock, fl.bodyBlock, fl.doneBlock)
	if entryPred == nil {
		// Cannot identify entry predecessor; skip transformation.
		return
	}

	// Step B: Create the break mask phi at the loop header.
	// break_mask = phi [entry: zero_mask, back_edge: new_break_mask]
	zeroMask := NewConst(constant.MakeBool(false), spmdpkg.NewVaryingMask())
	breakMaskPhi := &Phi{Comment: "spmd.break.mask"}
	breakMaskPhi.setType(spmdpkg.NewVaryingMask())
	breakMaskPhi.Edges = make([]Value, len(loopBlock.Preds))
	for i := range loopBlock.Preds {
		breakMaskPhi.Edges[i] = zeroMask // placeholder; entry and back-edge both start as zero
	}
	spmdInsertPhiAtFront(loopBlock, breakMaskPhi)
	spmdAddReferrer(zeroMask, breakMaskPhi) // entry edge

	// SPMDRegularBreaks is populated AFTER Step D (break linearization), because the
	// key must be the block containing the loop counter If AFTER CFG rewiring.
	// We update the map at the end of the function after lastElseBlock is known.

	// Step C: Snapshot result phis at doneBlock before any CFG rewiring.
	// We need the edge values (from then-blocks) before removePred compacts them.
	type resultPhiInfo struct {
		phi      *Phi
		breakVal map[*BasicBlock]Value // break-block → break value
		normVal  Value                 // value from the normal (non-break) loop exit
	}
	var resultPhis []resultPhiInfo
	for _, instr := range fl.doneBlock.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break
		}
		info := resultPhiInfo{
			phi:      phi,
			breakVal: make(map[*BasicBlock]Value),
		}
		for j, pred := range fl.doneBlock.Preds {
			isBreakPred := false
			for _, vb := range breaks {
				if pred == vb.thenBlock {
					info.breakVal[pred] = phi.Edges[j]
					isBreakPred = true
					break
				}
			}
			if !isBreakPred && info.normVal == nil {
				info.normVal = phi.Edges[j]
			}
		}
		resultPhis = append(resultPhis, info)
	}

	// Step B2: Create result accumulator phis at the loop header.
	// These carry the accumulated break result across iterations so that each
	// SPMDSelect merges new break values with the prior accumulated result,
	// rather than always falling back to the static default (normVal).
	//
	// Without these phis, runningAccum[i] would start as normVal each iteration
	// (because the SPMDSelect lives in ifBlock, not the loop header), causing
	// break results from earlier iterations to be discarded.
	//
	//   loopBlock:
	//       result_accum = phi [entry: normVal, back_edge: sel_result]
	accums := make([]*Phi, len(resultPhis))
	for i, rpi := range resultPhis {
		if len(rpi.breakVal) == 0 {
			continue
		}
		if rpi.normVal == nil {
			// All doneBlock predecessors are break-blocks; no normal-exit value
			// exists to seed the accumulator. Skip creating the phi.
			continue
		}
		accumPhi := &Phi{Comment: "spmd.break.accum"}
		accumPhi.setType(rpi.phi.Type())
		accumPhi.Edges = make([]Value, len(loopBlock.Preds))
		for j := range loopBlock.Preds {
			// All edges start as normVal: entry edge is the real value; back-edge
			// is a placeholder that will be replaced in Step E2 below.
			accumPhi.Edges[j] = rpi.normVal
			spmdAddReferrer(rpi.normVal, accumPhi)
		}
		spmdInsertPhiAtFront(loopBlock, accumPhi)
		accums[i] = accumPhi
	}

	// Step D: For each varying break, insert mask computation and SPMDSelect,
	// then linearize the If → Jump.
	//
	// The SPMDSelect is placed in ifBlock (inside the loop body). Its result
	// is passed out of the loop by updating the doneBlock phi's loop-exit edge.
	// This preserves the done block phi's entry edge (for the n=0 case), while
	// the loop-exit edge carries the accumulated break result.
	var lastBreakMask Value = breakMaskPhi
	// runningAccum[i] is the accumulated break result for resultPhis[i].
	// Initialized from the accumulator phi (loop-carried) so that SPMDSelect
	// merges with the prior iteration's result rather than the static default.
	runningAccum := make([]Value, len(resultPhis))
	for i, rpi := range resultPhis {
		if accums[i] != nil {
			runningAccum[i] = accums[i] // loop-carried accumulator
		} else {
			runningAccum[i] = rpi.normVal
		}
	}

	// Track the last elseBlock — after all breaks are linearized, this is the block
	// containing the loop counter If (the block TinyGo's *ssa.If handler will visit).
	var lastElseBlock *BasicBlock

	for _, vb := range breaks {
		ifBlock := vb.ifBlock
		thenBlock := vb.thenBlock
		elseBlock := vb.elseBlock
		lastElseBlock = elseBlock

		// Relocate FIRST: move thenBlock's non-terminator instructions into
		// ifBlock before inserting mask ops and SPMDSelect. This ensures values
		// defined in thenBlock (e.g., ChangeType for break-value assignments)
		// appear before the SPMDSelect that references them, maintaining the
		// producer-before-consumer ordering that TinyGo requires.
		spmdRelocateToBlock(thenBlock, ifBlock)

		// Compute active mask: AND_NOT(entryMask, lastBreakMask)
		breakActiveMask := spmdInsertMaskAndNot(ifBlock, activeMask, lastBreakMask)

		// Convert the condition to a mask and compute which lanes break.
		maskCond := spmdInsertConvertToMask(ifBlock, vb.vif.Cond)
		breakLanes := spmdInsertMaskAnd(ifBlock, breakActiveMask, maskCond)

		// Accumulate break mask.
		newBreakMask := spmdInsertMaskOr(ifBlock, lastBreakMask, breakLanes)
		lastBreakMask = newBreakMask

		// For each result phi that has a break value from this break block,
		// create SPMDSelect: sel = select(breakLanes, break_val, prev_accum).
		for i, rpi := range resultPhis {
			breakVal, hasBreakVal := rpi.breakVal[thenBlock]
			if !hasBreakVal {
				continue
			}
			sel := &SPMDSelect{
				Mask:  breakLanes,
				X:     breakVal,
				Y:     runningAccum[i],
				Lanes: lanes,
			}
			sel.setType(rpi.phi.Type())
			sel.setBlock(ifBlock)
			spmdInsertBeforeTerminator(ifBlock, sel)
			spmdAddReferrer(breakLanes, sel)
			spmdAddReferrer(breakVal, sel)
			spmdAddReferrer(runningAccum[i], sel)
			runningAccum[i] = sel
		}

		// Linearize: replace the varying If with a Jump to elseBlock.
		spmdReplaceIfWithJump(ifBlock, elseBlock, thenBlock)

		// Detach thenBlock from doneBlock (instrs already relocated above).
		fl.doneBlock.removePred(thenBlock)
		thenBlock.Succs = nil
		thenBlock.Instrs = nil // mark as dead
	}

	// Step E: Close the break mask phi's back-edge.
	// The back-edge is the predecessor that is NOT the entry predecessor.
	for i, pred := range loopBlock.Preds {
		if pred == entryPred {
			continue
		}
		// Remove the placeholder zero referrer and install the real value.
		if refs := zeroMask.Referrers(); refs != nil {
			*refs = removeInstr(*refs, breakMaskPhi)
		}
		breakMaskPhi.Edges[i] = lastBreakMask
		spmdAddReferrer(lastBreakMask, breakMaskPhi)
	}

	// Step E2: Close result accumulator phi back-edges.
	// Replace the placeholder normVal back-edge with the final SPMDSelect result
	// (runningAccum[i]), which is the value that should be carried into the next
	// loop iteration as the accumulated break result.
	for i, acc := range accums {
		if acc == nil {
			continue
		}
		for j, pred := range loopBlock.Preds {
			if pred == entryPred {
				continue // keep the entry edge as normVal
			}
			// Remove the placeholder normVal referrer for the back-edge and
			// install the real accumulated value (last SPMDSelect in loop body).
			if refs := resultPhis[i].normVal.Referrers(); refs != nil {
				*refs = spmdRemoveOneReferrer(*refs, acc)
			}
			acc.Edges[j] = runningAccum[i]
			spmdAddReferrer(runningAccum[i], acc)
		}
	}

	// Step F: Update doneBlock result phis to carry the accumulated break result.
	//
	// Instead of replacing the done block phi entirely (which would break the
	// entry → done path when n=0), we update the phi's loop-exit edge to point
	// to runningAccum[i] (the last SPMDSelect in the loop body).
	//
	// The loop-exit edge predecessor is the block that exits the loop to done.
	// That is: any predecessor of doneBlock that is NOT the entryPred and is NOT
	// a thenBlock (break block, already removed). After removing break predecessors
	// in Step D, the remaining doneBlock predecessors are: entryPred (direct entry
	// when n=0) and the loop-exit block (the block that exits via bound check).
	for i, rpi := range resultPhis {
		if runningAccum[i] == rpi.normVal {
			// No SPMDSelect was created for this phi; nothing to update.
			continue
		}
		// Update the done block phi's loop-exit edge (non-entry predecessor)
		// to use the accumulated break result.
		for j, pred := range fl.doneBlock.Preds {
			if pred == entryPred {
				continue // keep the entry edge as normVal
			}
			// This is the loop-exit edge. Update it to the accumulated result.
			// Use spmdRemoveOneReferrer to remove exactly one occurrence of the
			// old edge value from its referrer list. The value may appear in
			// other edges of this phi (e.g., entry edge), so we must not remove
			// all occurrences.
			if j < len(rpi.phi.Edges) {
				oldEdge := rpi.phi.Edges[j]
				if oldEdge != nil {
					if refs := oldEdge.Referrers(); refs != nil {
						*refs = spmdRemoveOneReferrer(*refs, rpi.phi)
					}
				}
				rpi.phi.Edges[j] = runningAccum[i]
				spmdAddReferrer(runningAccum[i], rpi.phi)
			}
		}
	}

	// Step G: Register the break mask phi in the function so TinyGo can emit an
	// early-exit check (all-lanes-broken → skip remaining iterations) at the
	// loop counter If. After CFG linearization in Step D, the loop counter If
	// lives in lastElseBlock (the continuation after all breaks). Use lastElseBlock
	// as the key so the TinyGo *ssa.If handler can find it via instr.Block().Index.
	if lastElseBlock != nil {
		if fn.SPMDRegularBreaks == nil {
			fn.SPMDRegularBreaks = make(map[*BasicBlock]*Phi)
		}
		fn.SPMDRegularBreaks[lastElseBlock] = breakMaskPhi
	}
}

// spmdLoopHeaderPreds identifies the entry predecessor and back-edge predecessor
// of a loop header block.
// The entry predecessor is the one that is NOT in the loop body (i.e., it does
// not reach loopBlock through a path that goes through bodyBlock or loopBlock).
// Returns (entryPred, backEdgePred). Returns (nil, nil) if ambiguous.
func spmdLoopHeaderPreds(loopBlock, bodyBlock, doneBlock *BasicBlock) (entryPred, backEdge *BasicBlock) {
	// Build the set of blocks reachable from loopBlock (the loop body).
	// A back-edge predecessor is one that is reachable from loopBlock itself.
	reachable := make(map[*BasicBlock]bool)
	queue := []*BasicBlock{loopBlock}
	reachable[loopBlock] = true
	for len(queue) > 0 {
		b := queue[0]
		queue = queue[1:]
		for _, succ := range b.Succs {
			if succ == doneBlock || reachable[succ] {
				continue
			}
			reachable[succ] = true
			queue = append(queue, succ)
		}
	}

	for _, pred := range loopBlock.Preds {
		if reachable[pred] {
			backEdge = pred
		} else {
			entryPred = pred
		}
	}
	return entryPred, backEdge
}

// spmdRemoveOneReferrer removes exactly one occurrence of instr from refs.
// Unlike removeInstr (which removes ALL occurrences), this is used when a
// value appears multiple times as an operand but only one occurrence is being
// replaced. Returns the updated slice.
func spmdRemoveOneReferrer(refs []Instruction, instr Instruction) []Instruction {
	for i, r := range refs {
		if r == instr {
			return slices.Delete(refs, i, i+1)
		}
	}
	return refs
}

// spmdInsertPhiAtFront inserts phi at the very beginning of block b's Instrs,
// before any existing instructions (including existing phis).
func spmdInsertPhiAtFront(b *BasicBlock, phi *Phi) {
	phi.setBlock(b)
	b.Instrs = append(b.Instrs, nil)
	copy(b.Instrs[1:], b.Instrs)
	b.Instrs[0] = phi
}

// spmdCompactInstrs removes nil slots from b.Instrs.
// Used after replacing instructions that leave nil gaps.
func spmdCompactInstrs(b *BasicBlock) {
	j := 0
	for _, instr := range b.Instrs {
		if instr != nil {
			b.Instrs[j] = instr
			j++
		}
	}
	for k := j; k < len(b.Instrs); k++ {
		b.Instrs[k] = nil // clear for GC
	}
	b.Instrs = b.Instrs[:j]
}

// spmdBuildExcludedIfs builds a set of If instructions that should NOT be
// processed by the generic varying-If predication pass. This includes:
//   - Ifs that are part of SPMDBooleanChains (short-circuit &&/||)
//   - Ifs that are part of SPMDSwitchChains (handled by predicateVaryingSwitch)
//
// Switch-chain Ifs are excluded here so the generic pass does not try to
// linearize individual case Ifs independently. predicateVaryingSwitch handles
// the entire chain as a unit.
func spmdBuildExcludedIfs(fn *Function) map[*If]bool {
	excluded := make(map[*If]bool)

	// Exclude switch-chain Ifs so predicateVaryingSwitch can handle them.
	for _, chain := range fn.SPMDSwitchChains {
		for _, caseIf := range chain.Cases {
			excluded[caseIf] = true
		}
	}

	for _, chain := range fn.SPMDBooleanChains {
		for _, block := range chain.Blocks {
			if len(block.Instrs) > 0 {
				if vif, ok := block.Instrs[len(block.Instrs)-1].(*If); ok {
					excluded[vif] = true
				}
			}
		}
	}

	return excluded
}

// spmdLoopScopeBlocks returns the set of blocks within the SPMD loop body.
// A block is in scope if it lies between the body block (inclusive) and the
// done block (exclusive), reachable via CFG traversal within the loop.
func spmdLoopScopeBlocks(loop *SPMDLoopInfo) map[*BasicBlock]bool {
	scope := make(map[*BasicBlock]bool)
	// BFS from BodyBlock (and LoopBlock if separate), stopping at DoneBlock.
	queue := []*BasicBlock{loop.BodyBlock}
	scope[loop.BodyBlock] = true
	if loop.LoopBlock != loop.BodyBlock {
		queue = append(queue, loop.LoopBlock)
		scope[loop.LoopBlock] = true
	}

	for len(queue) > 0 {
		b := queue[0]
		queue = queue[1:]
		for _, succ := range b.Succs {
			if succ == loop.DoneBlock || scope[succ] {
				continue
			}
			// Stop at the entry block to avoid going backwards.
			if succ == loop.EntryBlock {
				continue
			}
			scope[succ] = true
			queue = append(queue, succ)
		}
	}

	return scope
}

// spmdTailScopeBlocks returns the subset of liveScopeBlocks that belong to the
// tail phase of a peeled SPMD loop. For non-peeled loops, returns an empty map.
// Identifies tail blocks by BFS from TailBodyBlock bounded by DoneBlock and
// TrampolineBlock to avoid crossing into the main phase or post-loop blocks.
func spmdTailScopeBlocks(loop *SPMDLoopInfo, liveScopeBlocks map[*BasicBlock]bool) map[*BasicBlock]bool {
	tail := make(map[*BasicBlock]bool)
	if !loop.IsPeeled || loop.TailBodyBlock == nil {
		return tail
	}
	if !liveScopeBlocks[loop.TailBodyBlock] {
		return tail
	}
	queue := []*BasicBlock{loop.TailBodyBlock}
	tail[loop.TailBodyBlock] = true
	// Include TailCheckBlock if it is in scope.
	if loop.TailCheckBlock != nil && liveScopeBlocks[loop.TailCheckBlock] {
		tail[loop.TailCheckBlock] = true
		queue = append(queue, loop.TailCheckBlock)
	}
	for len(queue) > 0 {
		b := queue[0]
		queue = queue[1:]
		for _, succ := range b.Succs {
			if tail[succ] || !liveScopeBlocks[succ] {
				continue
			}
			// Stop at phase and post-loop boundaries.
			if succ == loop.DoneBlock || succ == loop.TrampolineBlock {
				continue
			}
			if succ == loop.MainBodyBlock {
				continue
			}
			tail[succ] = true
			queue = append(queue, succ)
		}
	}
	return tail
}

// predicateVaryingIf linearizes a single varying If instruction.
//
// Standard pattern (simple diamond or if-without-else):
//
//	B: ... if cond goto T else E (IsVarying)
//	T: ... [some values] ... jump M
//	E: ... [some values] ... jump M
//	M: phi [T: v1, E: v2] ...
//
// Transforms to:
//
//	B: ... mask_cond = convert(cond); then_mask = active_mask & mask_cond; else_mask = active_mask &^ mask_cond; jump T
//	T: ... [SPMDLoad/SPMDStore with then_mask] ... jump E
//	E: ... [SPMDLoad/SPMDStore with else_mask] ... jump M
//	M: SPMDSelect(then_mask, v1, v2) replaces phi [T: v1, E: v2] ...
//
// deferred, when non-nil (go-for loop context), receives loop-header merge
// records when both then and else branches jump to the same loop header.
// The If is linearized (B→T→E→loopHeader) and phi edge values are snapshotted,
// but phi→SPMDSelect conversion is deferred to spmdConvertDeferredMerges so that
// the loop header's pred list is fully stable before any SPMDSelect is inserted.
func predicateVaryingIf(fn *Function, lanes int, ifBlock *BasicBlock, vif *If, allowLoopHeaderMerge bool, activeMask Value, deferred *[]*spmdDeferredMerge) {
	thenBlock := ifBlock.Succs[0] // If.Cond true → Succs[0]
	elseBlock := ifBlock.Succs[1] // If.Cond false → Succs[1]

	// Find the merge (done) block — the block both paths converge to.
	// Returns nil for complex patterns (compound booleans, multi-block chains).
	mergeBlock := findMergeBlock(thenBlock, elseBlock)
	if mergeBlock == nil {
		// LOR short-circuit pattern: elseBlock's sole successor is thenBlock.
		// This is logicalBinop(LOR) with a varying condition:
		//   B: if cond goto T(done) else E(rhs)
		//   E: compute rhs, goto T
		//   T: phi [B: true/short, E: rhs_val] ...
		// In SPMD, all lanes must execute E regardless of cond. Linearize B→E→T
		// and replace the phi at T with SPMDSelect(mask_cond, short_val, rhs_val).
		if len(elseBlock.Succs) == 1 && elseBlock.Succs[0] == thenBlock {
			spmdLinearizeLORShortCircuit(fn, ifBlock, vif, thenBlock, elseBlock, activeMask, lanes)
			return
		}

		// Check for loop-header merge pattern: both branches converge
		// to the same block but it has >2 predecessors (loop header).
		if allowLoopHeaderMerge &&
			len(thenBlock.Succs) == 1 && len(elseBlock.Succs) == 1 &&
			thenBlock.Succs[0] == elseBlock.Succs[0] {
			// SPMD function body context: insert a trampoline merge block to
			// create a proper 2-predecessor diamond. This is safe here because
			// there are no SSA-level loop peeling operations that could be broken.
			loopHeader := thenBlock.Succs[0]
			mergeBlock = spmdInsertMergeTrampoline(fn, thenBlock, elseBlock, loopHeader)
		} else if !allowLoopHeaderMerge && deferred != nil &&
			len(thenBlock.Succs) == 1 && len(elseBlock.Succs) == 1 &&
			thenBlock.Succs[0] == elseBlock.Succs[0] {
			// Go-for loop context: both branches jump to the loop header.
			// We cannot insert a trampoline (would break peelSPMDLoop's structural
			// expectations). Instead, linearize B→T→E→loopHeader and defer the
			// phi→SPMDSelect conversion until after all varying Ifs are processed.
			spmdLinearizeLoopHeaderMerge(fn, ifBlock, vif, thenBlock, elseBlock, activeMask, lanes, deferred)
			return
		}
		if mergeBlock == nil {
			return // genuinely complex pattern, skip
		}
	}

	// Build mask instructions: insert before the If terminator in ifBlock.
	maskCond := spmdInsertConvertToMask(ifBlock, vif.Cond)
	thenMask := spmdInsertMaskAnd(ifBlock, activeMask, maskCond)

	var elseMask Value
	ifWithoutElse := elseBlock == mergeBlock
	if !ifWithoutElse {
		elseMask = spmdInsertMaskAndNot(ifBlock, activeMask, maskCond)
	}

	// Replace the If terminator with a Jump to thenBlock.
	spmdReplaceIfWithJump(ifBlock, thenBlock, elseBlock)

	if ifWithoutElse {
		// if-without-else: B→T→M (E is M)
		// The then-block already jumps to M. No CFG rewiring needed,
		// but we must fix the Phi edges at M since B no longer flows to M directly.
		// After linearization, M's predecessor is T only (B→T→M).
		// But wait: with if-without-else, thenBlock jumps to elseBlock (==mergeBlock).
		// The merge still has T as predecessor (via the original T→M jump), so no
		// Phi edge changes are needed.
		// Replace Phis at merge with SPMDSelect using thenMask.
		spmdReplacePhisWithSelect(mergeBlock, thenBlock, ifBlock, thenMask, thenMask, lanes)

		// Replace loads/stores in then-block.
		spmdMaskMemOps(thenBlock, thenMask, lanes)
	} else {
		// if-else: B→T→E→M
		// Rewire: thenBlock's successor from M → E.
		spmdRewireThenToElse(thenBlock, mergeBlock, elseBlock)

		// Replace Phis at merge with SPMDSelect.
		spmdReplacePhisWithSelect(mergeBlock, thenBlock, elseBlock, thenMask, elseMask, lanes)

		// Replace loads/stores in both branches.
		spmdMaskMemOps(thenBlock, thenMask, lanes)
		spmdMaskMemOps(elseBlock, elseMask, lanes)
	}
}

// spmdLinearizeLoopHeaderMerge handles the go-for loop case where both then and
// else branches of a varying If jump to the same loop header. We cannot insert
// a trampoline (that would break peelSPMDLoop's block-structure expectations),
// so instead we:
//  1. Snapshot phi edges at the loop header before any CFG modification.
//  2. Compute then/else masks, insert them before the If terminator.
//  3. Replace the If with a Jump to thenBlock (B→T).
//  4. Rewire thenBlock→loopHeader to thenBlock→elseBlock (B→T→E→loopHeader).
//  5. Mask memory ops in both branches.
//  6. Record a spmdDeferredMerge for phi→SPMDSelect conversion after all
//     varying Ifs in the scope have been processed.
//
// The phi→SPMDSelect conversion cannot happen here because the loop header may
// gain or lose other predecessors as sibling varying Ifs are linearized in the
// same scope pass.
func spmdLinearizeLoopHeaderMerge(fn *Function, ifBlock *BasicBlock, vif *If, thenBlock, elseBlock *BasicBlock, activeMask Value, lanes int, deferred *[]*spmdDeferredMerge) {
	loopHeader := thenBlock.Succs[0]

	// Step 1: Snapshot phi edges at loopHeader BEFORE any CFG rewiring.
	// After spmdRewireThenToElse, thenBlock is removed from loopHeader.Preds
	// and the corresponding Phi edges are compacted away. We need those edge
	// values to build the SPMDSelect instructions later.
	var phiSnaps []spmdPhiSnapshot
	for _, instr := range loopHeader.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break // phis are always first
		}
		snap := spmdPhiSnapshot{phi: phi, edgeVal: make(map[*BasicBlock]Value, len(loopHeader.Preds))}
		for j, pred := range loopHeader.Preds {
			snap.edgeVal[pred] = phi.Edges[j]
		}
		phiSnaps = append(phiSnaps, snap)
	}

	// Step 2: Compute masks (insert before the If terminator in ifBlock).
	maskCond := spmdInsertConvertToMask(ifBlock, vif.Cond)
	thenMask := spmdInsertMaskAnd(ifBlock, activeMask, maskCond)
	elseMask := spmdInsertMaskAndNot(ifBlock, activeMask, maskCond)

	// Step 3: Replace the If with a Jump to thenBlock. Removes ifBlock→elseBlock edge.
	spmdReplaceIfWithJump(ifBlock, thenBlock, elseBlock)

	// Step 4: Rewire thenBlock→loopHeader to thenBlock→elseBlock.
	// spmdRewireThenToElse updates thenBlock.Succs and adds thenBlock to
	// elseBlock.Preds, but intentionally does NOT call loopHeader.removePred.
	// (The normal non-deferred path relies on spmdReplacePhisWithSelect doing that.)
	// Here we must explicitly remove thenBlock from loopHeader.Preds and compact
	// the phi edges. The phi edge values were already snapshotted above.
	spmdRewireThenToElse(thenBlock, loopHeader, elseBlock)
	loopHeader.removePred(thenBlock)

	// Step 5: Mask memory ops in both branches.
	spmdMaskMemOps(thenBlock, thenMask, lanes)
	spmdMaskMemOps(elseBlock, elseMask, lanes)

	// Step 6: Record deferred merge for phi→SPMDSelect conversion.
	*deferred = append(*deferred, &spmdDeferredMerge{
		thenBlock:  thenBlock,
		elseBlock:  elseBlock,
		mergeBlock: loopHeader,
		thenMask:   thenMask,
		elseMask:   elseMask,
		lanes:      lanes,
		phiSnaps:   phiSnaps,
	})
}

// spmdLinearizeLORShortCircuit handles a varying If generated by logicalBinop(LOR)
// when used as a reified value (not as a direct if-condition). The SSA pattern is:
//
//	B:  if cond goto T(done) else E(rhs)    [LOR short-circuit]
//	E:  compute rhs, goto T
//	T:  phi [B: short_val, E: rhs_val] ...  [binop.done merge phi]
//
// In SPMD, all lanes must execute E regardless of cond (no per-lane branching).
// We linearize to B→E→T and replace the phi at T with SPMDSelect(mask_cond,
// short_val, rhs_val), giving each lane the correct per-lane result.
func spmdLinearizeLORShortCircuit(fn *Function, ifBlock *BasicBlock, vif *If, thenBlock, elseBlock *BasicBlock, activeMask Value, lanes int) {
	// thenBlock = binop.done (the merge), elseBlock = binop.rhs (the rhs block).
	// elseBlock's sole successor is thenBlock.

	// Step 1: Snapshot phi edges at thenBlock (the merge) BEFORE CFG rewiring.
	// spmdReplaceIfWithJump below removes ifBlock from thenBlock.Preds and
	// compacts phi edges. We need the original edges to build SPMDSelect.
	var phiSnaps []spmdPhiSnapshot
	for _, instr := range thenBlock.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break // phis are always first
		}
		snap := spmdPhiSnapshot{phi: phi, edgeVal: make(map[*BasicBlock]Value, len(thenBlock.Preds))}
		for j, pred := range thenBlock.Preds {
			snap.edgeVal[pred] = phi.Edges[j]
		}
		phiSnaps = append(phiSnaps, snap)
	}

	// Step 2: Compute the condition mask (insert before the If in ifBlock).
	maskCond := spmdInsertConvertToMask(ifBlock, vif.Cond)
	thenMask := spmdInsertMaskAnd(ifBlock, activeMask, maskCond)

	// Step 3: Replace the If in ifBlock with Jump to elseBlock (rhs).
	// This linearizes B→E instead of B→T (short-circuit). The If's edge to
	// thenBlock is removed: ifBlock is removed from thenBlock.Preds.
	spmdReplaceIfWithJump(ifBlock, elseBlock, thenBlock)

	// Step 4: Convert phis at thenBlock to SPMDSelect.
	// short_val = snapshotted edge from ifBlock (the short-circuit constant, e.g. true)
	// rhs_val   = snapshotted edge from elseBlock (the computed RHS value)
	// SPMDSelect(thenMask, short_val, rhs_val): lanes where cond was true get
	// short_val; other lanes get the freshly computed rhs_val.
	spmdReplaceBooleanChainPhis(thenBlock, ifBlock, elseBlock, thenMask, lanes, phiSnaps)

	// Step 5: Mask mem ops in elseBlock. Unlike a standard varying-if (which uses
	// elseMask = activeMask & ~cond), the LOR short-circuit always evaluates the
	// RHS for all active lanes to get per-lane results. So we use activeMask here.
	// In practice elseBlock typically contains only arithmetic, so this is a no-op.
	spmdMaskMemOps(elseBlock, activeMask, lanes)
}

// spmdConvertDeferredMerges processes loop-header merge records collected during
// predicateSPMDScope. For each record, it inserts an SPMDSelect in elseBlock for
// every loop-header phi that had different values from thenBlock and elseBlock,
// and redirects the phi's elseBlock edge to the SPMDSelect result.
//
// After spmdLinearizeLoopHeaderMerge ran:
//   - loopHeader lost thenBlock as a predecessor (its phi edges were compacted).
//   - elseBlock still flows to loopHeader, so values placed in elseBlock dominate
//     loopHeader.
//   - phiSnaps holds the original thenBlock and elseBlock edge values for each phi.
//
// For each phi at loopHeader with snapshotted then/else edge values:
//   - If thenVal == elseVal: no select needed; the existing edge is already correct.
//   - Otherwise: insert SPMDSelect(thenMask, thenVal, elseVal) in elseBlock just
//     before its terminator, and update the phi's elseBlock edge to reference it.
func spmdConvertDeferredMerges(fn *Function, deferred []*spmdDeferredMerge) {
	for _, dm := range deferred {
		for _, snap := range dm.phiSnaps {
			thenVal, hasThen := snap.edgeVal[dm.thenBlock]
			elseVal, hasElse := snap.edgeVal[dm.elseBlock]
			if !hasThen || !hasElse {
				// Phi was not driven by both branches; nothing to do.
				continue
			}
			if thenVal == elseVal {
				// Both branches carried the same value; no select needed.
				continue
			}

			// Create SPMDSelect: select(thenMask, thenVal, elseVal).
			// The result type matches the phi type.
			sel := &SPMDSelect{
				Mask:  dm.thenMask,
				X:     thenVal,
				Y:     elseVal,
				Lanes: dm.lanes,
			}
			sel.setType(snap.phi.Type())

			// Insert SPMDSelect in elseBlock just before its terminator.
			// elseBlock still flows to loopHeader, so the select dominates the phi.
			spmdInsertBeforeTerminator(dm.elseBlock, sel)
			spmdAddReferrer(dm.thenMask, sel)
			spmdAddReferrer(thenVal, sel)
			spmdAddReferrer(elseVal, sel)

			// Update the phi's elseBlock edge to reference the SPMDSelect result.
			// The phi.Edges slice was compacted when thenBlock was removed, so we
			// must find the current position of elseBlock in mergeBlock.Preds.
			for j, pred := range dm.mergeBlock.Preds {
				if pred == dm.elseBlock {
					oldEdge := snap.phi.Edges[j]
					if oldEdge != nil {
						if refs := oldEdge.Referrers(); refs != nil {
							*refs = spmdRemoveOneReferrer(*refs, snap.phi)
						}
					}
					snap.phi.Edges[j] = sel
					spmdAddReferrer(sel, snap.phi)
					break
				}
			}
		}
	}
}

// spmdConvertScopedMemOps converts UnOp{MUL} (pointer loads) and Store
// instructions in scopeBlocks to SPMDLoad/SPMDStore with the given active mask.
// This is the scoped variant of spmdConvertAllMemOps for go-for loop bodies.
//
// Unlike spmdConvertAllMemOps (used for func bodies), this function includes
// contiguity detection via spmdIsContiguousIndex: loads/stores whose address is
// an IndexAddr with an iter-based index are marked Contiguous so TinyGo can emit
// a single contiguous load/store instruction instead of a gather/scatter.
//
// Instructions already converted to SPMDLoad/SPMDStore (by spmdMaskMemOps inside
// varying branches) are not present as UnOp/Store and are therefore skipped.
func spmdConvertScopedMemOps(fn *Function, scopeBlocks map[*BasicBlock]bool, mask Value, lanes int) {
	for _, block := range fn.Blocks {
		if !scopeBlocks[block] {
			continue
		}
		for i := 0; i < len(block.Instrs); i++ {
			instr := block.Instrs[i]

			switch instr := instr.(type) {
			case *UnOp:
				if instr.Op != token.MUL {
					continue
				}
				// Only convert loads of SPMD-compatible element types.
				if !spmdIsVectorizableElemType(instr.Type()) {
					continue
				}
				load := &SPMDLoad{
					Addr:  instr.X,
					Mask:  mask,
					Lanes: lanes,
					pos:   instr.Pos(),
				}
				// Detect contiguous access: IndexAddr with iter-based index.
				if indexAddr, ok := instr.X.(*IndexAddr); ok {
					if spmdIsContiguousIndex(fn, indexAddr.Index) {
						load.Contiguous = true
						load.Source = indexAddr.X
					}
				}
				load.setType(instr.Type())
				load.setBlock(block)

				spmdAddReferrer(instr.X, load)
				spmdAddReferrer(mask, load)
				if load.Source != nil {
					spmdAddReferrer(load.Source, load)
				}

				block.Instrs[i] = load
				replaceAll(instr, load)

				if refs := instr.X.Referrers(); refs != nil {
					*refs = removeInstr(*refs, instr)
				}
				instr.block = nil

			case *Store:
				// Only convert stores of SPMD-compatible element types.
				if !spmdIsVectorizableElemType(instr.Val.Type()) {
					continue
				}
				store := &SPMDStore{
					Addr:  instr.Addr,
					Val:   instr.Val,
					Mask:  mask,
					Lanes: lanes,
					pos:   instr.Pos(),
				}
				// Detect contiguous access: IndexAddr with iter-based index.
				if indexAddr, ok := instr.Addr.(*IndexAddr); ok {
					if spmdIsContiguousIndex(fn, indexAddr.Index) {
						store.Contiguous = true
						store.Source = indexAddr.X
					}
				}
				store.setBlock(block)

				spmdAddReferrer(instr.Addr, store)
				spmdAddReferrer(instr.Val, store)
				spmdAddReferrer(mask, store)
				if store.Source != nil {
					spmdAddReferrer(store.Source, store)
				}

				block.Instrs[i] = store

				if refs := instr.Addr.Referrers(); refs != nil {
					*refs = removeInstr(*refs, instr)
				}
				if refs := instr.Val.Referrers(); refs != nil {
					*refs = removeInstr(*refs, instr)
				}
				instr.block = nil
			}
		}
	}
}

// spmdMaskScopedCallOps sweeps Call instructions in scopeBlocks and sets
// SPMDMask = defaultMask for any SPMD function call that does not already have
// a mask assigned (i.e., calls in straight-line code that were not inside a
// varying if/else/switch, which already received narrowed masks via spmdMaskMemOps).
func spmdMaskScopedCallOps(fn *Function, scopeBlocks map[*BasicBlock]bool, defaultMask Value) {
	for _, block := range fn.Blocks {
		if !scopeBlocks[block] {
			continue
		}
		for _, instr := range block.Instrs {
			call, ok := instr.(*Call)
			if !ok || call.Call.SPMDMask != nil {
				continue
			}
			callee := call.Call.StaticCallee()
			if callee == nil || !hasSPMDParams(callee) {
				continue
			}
			call.Call.SPMDMask = defaultMask
			spmdAddReferrer(defaultMask, call)
		}
	}
}

// spmdMaskScopedIndexOps sets SPMDMask on IndexAddr/Index instructions with
// varying (SPMDType) indices in scopeBlocks. Skips instructions that already
// have SPMDMask set (e.g., by spmdMaskMemOps for varying-if blocks, which
// narrowed the mask to the then/else sub-mask).
func spmdMaskScopedIndexOps(fn *Function, scopeBlocks map[*BasicBlock]bool, mask Value) {
	for _, block := range fn.Blocks {
		if !scopeBlocks[block] {
			continue
		}
		for _, instr := range block.Instrs {
			switch v := instr.(type) {
			case *IndexAddr:
				if v.SPMDMask != nil {
					continue
				}
				if _, ok := v.Index.Type().(*types.SPMDType); ok {
					v.SPMDMask = mask
					spmdAddReferrer(mask, v)
				}
			case *Index:
				if v.SPMDMask != nil {
					continue
				}
				if _, ok := v.Index.Type().(*types.SPMDType); ok {
					v.SPMDMask = mask
					spmdAddReferrer(mask, v)
				}
			}
		}
	}
}

// findMergeBlock returns the block where the then-path and else-path converge,
// or nil if the pattern is not a simple if/else or if-without-else diamond.
//
// Only two structural patterns are recognised:
//  1. Simple diamond: thenBlock and elseBlock each have exactly one successor
//     and they are the same block, AND that block has only those two as
//     predecessors (so it is a true merge, not a loop header).
//  2. If-without-else: thenBlock has exactly one successor which is elseBlock
//     (i.e., the "else" is a fall-through and elseBlock IS the merge).
//
// Any more complex pattern (compound boolean, multi-block chains, or a merge
// block that is also a loop header with extra predecessors) returns nil
// and the varying If will be skipped by the caller to avoid incorrect CFG
// rewiring. In particular, if both then/else jump back to a loop header, the
// merge block would be the loop header — which has additional predecessors
// (e.g., the loop entry) and whose Phis are loop-carried values, not if-else
// merge values. Replacing those Phis with SPMDSelect would be incorrect
// because the mask is defined inside the loop body (domination violation).
func findMergeBlock(thenBlock, elseBlock *BasicBlock) *BasicBlock {
	// Simple diamond: both branches have a single common successor.
	if len(thenBlock.Succs) == 1 && len(elseBlock.Succs) == 1 {
		merge := thenBlock.Succs[0]
		if merge == elseBlock.Succs[0] {
			// Guard: the merge block must have exactly two predecessors
			// (thenBlock and elseBlock). If it has more (e.g., a loop header
			// also reachable from the loop entry), it is not a safe merge
			// point for SPMDSelect because loop-carried Phis would be
			// incorrectly replaced, and the mask (computed in the loop body)
			// would not dominate uses in the loop header.
			if len(merge.Preds) != 2 {
				return nil
			}
			return merge
		}
	}

	// If-without-else: then-block's sole successor is the else-block (fall-through).
	if len(thenBlock.Succs) == 1 && thenBlock.Succs[0] == elseBlock {
		return elseBlock
	}

	// Complex pattern: cannot safely identify the merge block.
	// The caller must skip this varying If.
	return nil
}

// spmdInsertMergeTrampoline creates a new block between thenBlock/elseBlock and
// their shared successor (the loop header). It redirects both branches to the
// new merge block, which then jumps to the original successor. Phis at the
// successor that referenced thenBlock/elseBlock are updated to reference the
// merge block, with new phis in the merge block if the then/else values differ.
//
// This handles the "loop-header merge" pattern where a varying if/else inside a
// loop has both branches jumping back to the loop header. The loop header has
// 3+ predecessors (entry + then + else), so findMergeBlock rejects it. By
// inserting a trampoline, we create a proper 2-predecessor merge block that
// findMergeBlock's caller can predicate normally.
func spmdInsertMergeTrampoline(fn *Function, thenBlock, elseBlock, loopBlock *BasicBlock) *BasicBlock {
	// Step 1: Create new merge block.
	mergeBlock := fn.newBasicBlock("spmd.merge")

	// Step 2: Add Jump instruction in merge block → loopBlock.
	jmp := &Jump{}
	jmp.setBlock(mergeBlock)
	mergeBlock.Instrs = append(mergeBlock.Instrs, jmp)

	// Step 3: Rewire thenBlock and elseBlock successors from loopBlock → mergeBlock.
	thenBlock.replaceSucc(loopBlock, mergeBlock)
	elseBlock.replaceSucc(loopBlock, mergeBlock)

	// Step 4: Set up mergeBlock's predecessor/successor edges.
	mergeBlock.Preds = []*BasicBlock{thenBlock, elseBlock}
	addEdge(mergeBlock, loopBlock)
	// addEdge appends mergeBlock to loopBlock.Preds, but we still need to
	// remove thenBlock and elseBlock from loopBlock.Preds and fix phis.
	// We handle this below.

	// Step 5: Update loopBlock's phis and predecessor list.
	// For each phi in loopBlock, find the edges from thenBlock and elseBlock.
	// Replace both with a single edge from mergeBlock. If the values differ,
	// create a phi in mergeBlock.
	for _, instr := range loopBlock.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break // phis are always first
		}
		var thenVal, elseVal Value
		var thenIdx, elseIdx int = -1, -1
		for j, pred := range loopBlock.Preds {
			if pred == thenBlock {
				thenVal = phi.Edges[j]
				thenIdx = j
			} else if pred == elseBlock {
				elseVal = phi.Edges[j]
				elseIdx = j
			}
		}
		if thenIdx == -1 || elseIdx == -1 {
			// Phi doesn't reference both then/else — should not happen
			// in the expected pattern, but be defensive. Still grow
			// Edges to match the Preds growth from addEdge (Step 4).
			// Use the first edge value as a placeholder; Step 6's
			// spmdRemovePredAt will shrink it back to the correct length.
			phi.Edges = append(phi.Edges, phi.Edges[0])
			continue
		}

		// Determine the value for the mergeBlock edge.
		var mergeVal Value
		if thenVal == elseVal {
			// Both edges carry the same value — no new phi needed.
			// Remove one referrer entry: two edges (then+else) collapse
			// to one (merge), so net count should decrease by 1.
			mergeVal = thenVal
			if refs := mergeVal.Referrers(); refs != nil {
				*refs = removeInstr(*refs, phi)
			}
		} else {
			// Different values: create a phi in mergeBlock.
			mergePhi := &Phi{Comment: "spmd.merge"}
			mergePhi.setType(phi.Type())
			// mergeBlock.Preds = [thenBlock, elseBlock]
			mergePhi.Edges = []Value{thenVal, elseVal}
			spmdInsertPhiAtFront(mergeBlock, mergePhi)
			spmdAddReferrer(thenVal, mergePhi)
			spmdAddReferrer(elseVal, mergePhi)
			mergeVal = mergePhi
		}

		// The mergeBlock edge was added by addEdge at the end of loopBlock.Preds.
		// Set its phi edge value.
		mergeEdgeIdx := len(loopBlock.Preds) - 1
		phi.Edges = append(phi.Edges, nil) // grow to match Preds length
		phi.Edges[mergeEdgeIdx] = mergeVal
		spmdAddReferrer(mergeVal, phi)

		// Remove referrers for old then/else edge values that are no longer
		// referenced by this phi. When mergeVal is the same as thenVal/elseVal,
		// the phi still references that value through the merge edge, so we
		// must NOT remove its referrer entry (removeInstr removes ALL occurrences).
		if thenVal != mergeVal {
			if refs := thenVal.Referrers(); refs != nil {
				*refs = removeInstr(*refs, phi)
			}
		}
		if elseVal != mergeVal && elseVal != thenVal {
			if refs := elseVal.Referrers(); refs != nil {
				*refs = removeInstr(*refs, phi)
			}
		}
	}

	// Step 6: Remove thenBlock and elseBlock from loopBlock.Preds.
	// We must remove them carefully to preserve edge ordering for other phis.
	// Remove the higher index first to avoid shifting issues.
	// Find their indices in the current Preds list.
	var thenPredIdx, elsePredIdx int = -1, -1
	for i, pred := range loopBlock.Preds {
		if pred == thenBlock && thenPredIdx == -1 {
			thenPredIdx = i
		} else if pred == elseBlock && elsePredIdx == -1 {
			elsePredIdx = i
		}
	}

	if thenPredIdx == -1 || elsePredIdx == -1 {
		// Should never happen: the trampoline condition guarantees both
		// are predecessors of loopBlock.
		return mergeBlock
	}

	// Remove higher index first.
	if thenPredIdx > elsePredIdx {
		spmdRemovePredAt(loopBlock, thenPredIdx)
		spmdRemovePredAt(loopBlock, elsePredIdx)
	} else {
		spmdRemovePredAt(loopBlock, elsePredIdx)
		spmdRemovePredAt(loopBlock, thenPredIdx)
	}

	return mergeBlock
}

// spmdRemovePredAt removes the predecessor at index idx from block b,
// along with the corresponding phi edges. Does not use removePred because
// we need index-based removal to handle the case where we're removing
// two specific predecessors and need to preserve ordering.
func spmdRemovePredAt(b *BasicBlock, idx int) {
	// Remove from Preds.
	b.Preds = append(b.Preds[:idx], b.Preds[idx+1:]...)

	// Remove corresponding phi edges.
	for _, instr := range b.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break
		}
		phi.Edges = append(phi.Edges[:idx], phi.Edges[idx+1:]...)
	}
}

// spmdAllOnesMask returns a Const of type Varying[mask] representing all lanes active.
// This is the outermost active mask (no enclosing varying if).
func spmdAllOnesMask() Value {
	return NewConst(constant.MakeBool(true), spmdpkg.NewVaryingMask())
}

// spmdInsertConvertToMask inserts a Convert(cond, Varying[mask]) before the
// last instruction of block b. Returns the Convert value.
func spmdInsertConvertToMask(b *BasicBlock, cond Value) *Convert {
	c := &Convert{X: cond}
	c.setType(spmdpkg.NewVaryingMask())
	spmdInsertBeforeTerminator(b, c)
	spmdAddReferrer(cond, c)
	return c
}

// spmdInsertMaskAnd inserts a BinOp(AND, x, y) of Varying[mask] type before
// the last instruction of block b. Returns the BinOp value.
func spmdInsertMaskAnd(b *BasicBlock, x, y Value) *BinOp {
	op := &BinOp{Op: token.AND, X: x, Y: y}
	op.setType(spmdpkg.NewVaryingMask())
	spmdInsertBeforeTerminator(b, op)
	spmdAddReferrer(x, op)
	spmdAddReferrer(y, op)
	return op
}

// spmdInsertMaskAndNot inserts a BinOp(AND_NOT, x, y) of Varying[mask] type
// before the last instruction of block b. Returns the BinOp value.
func spmdInsertMaskAndNot(b *BasicBlock, x, y Value) *BinOp {
	op := &BinOp{Op: token.AND_NOT, X: x, Y: y}
	op.setType(spmdpkg.NewVaryingMask())
	spmdInsertBeforeTerminator(b, op)
	spmdAddReferrer(x, op)
	spmdAddReferrer(y, op)
	return op
}

// spmdInsertMaskOr inserts a BinOp(OR, x, y) of Varying[mask] type before
// the last instruction of block b. Returns the BinOp value.
func spmdInsertMaskOr(b *BasicBlock, x, y Value) *BinOp {
	op := &BinOp{Op: token.OR, X: x, Y: y}
	op.setType(spmdpkg.NewVaryingMask())
	spmdInsertBeforeTerminator(b, op)
	spmdAddReferrer(x, op)
	spmdAddReferrer(y, op)
	return op
}

// predicateBooleanChain linearizes a varying SPMDBooleanChain (&&/||) into a
// linear sequence of blocks that computes a combined condition mask, then
// applies standard predication (mask-gated memory ops, SPMDSelect at merge).
//
// For `if a && b { then } else { else_ }`:
//
//	B0: compute a; If a → B1 else ElseBlock  (LAND: false exits early)
//	B1: compute b; If b → ThenBlock else ElseBlock
//
// Transforms to:
//
//	B0: maskA=convert(a); combined=maskA; Jump B1
//	B1: maskB=convert(b); combined=AND(combined,maskB); thenMask=AND(active,combined); Jump ThenBlock
//	ThenBlock: [SPMDLoad/SPMDStore with thenMask] Jump ElseBlock
//	ElseBlock: [SPMDLoad/SPMDStore with elseMask] Jump Merge
//	Merge: SPMDSelect(thenMask, thenVal, elseVal)
//
// For `if a || b { then } else { else_ }`:
//
//	B0: compute a; If a → ThenBlock else B1  (LOR: true exits early)
//	B1: compute b; If b → ThenBlock else ElseBlock
//
// Same transformation: B0 → B1 → ThenBlock, combined=OR(maskA, maskB).
func predicateBooleanChain(fn *Function, lanes int, chain *SPMDBooleanChain, activeMask Value, deferred *[]*spmdDeferredMerge) {
	if len(chain.Blocks) < 2 {
		return
	}

	thenBlock := chain.ThenBlock
	elseBlock := chain.ElseBlock

	// Step 0: Find the merge block and snapshot Phi edges BEFORE any CFG
	// rewiring. spmdReplaceIfWithJump calls removePred which compacts
	// Phi.Edges, losing the "else" values needed for SPMDSelect.
	mergeBlock := findMergeBlock(thenBlock, elseBlock)

	// Detect loop-header merge: mergeBlock==nil when the shared successor has
	// 3+ predecessors (loop header case). Snapshot its phis now while the
	// CFG is still intact — chain block rewiring below does not touch the
	// loop header's Preds, so this snapshot captures the correct edge values.
	var loopHeaderSnaps []spmdPhiSnapshot
	if mergeBlock == nil && elseBlock != nil && len(thenBlock.Succs) == 1 && len(elseBlock.Succs) == 1 &&
		thenBlock.Succs[0] == elseBlock.Succs[0] {
		loopHeader := thenBlock.Succs[0]
		for _, instr := range loopHeader.Instrs {
			phi, ok := instr.(*Phi)
			if !ok {
				break
			}
			snap := spmdPhiSnapshot{phi: phi, edgeVal: make(map[*BasicBlock]Value, len(loopHeader.Preds))}
			for j, pred := range loopHeader.Preds {
				snap.edgeVal[pred] = phi.Edges[j]
			}
			loopHeaderSnaps = append(loopHeaderSnaps, snap)
		}
	}

	// Snapshot Phi edge values at the merge block. For if-without-else
	// (elseBlock == mergeBlock), chain blocks are predecessors of mergeBlock
	// and will be removed during linearization.
	var phiSnapshots []spmdPhiSnapshot
	if mergeBlock != nil {
		for _, instr := range mergeBlock.Instrs {
			phi, ok := instr.(*Phi)
			if !ok {
				break
			}
			edges := make(map[*BasicBlock]Value)
			for j, pred := range mergeBlock.Preds {
				edges[pred] = phi.Edges[j]
			}
			phiSnapshots = append(phiSnapshots, spmdPhiSnapshot{phi: phi, edgeVal: edges})
		}
	}

	// Step 1: For each chain block, convert its condition to a mask and
	// accumulate the combined mask. Insert all operations before the block's
	// If terminator. Because each chain block dominates the next (they form
	// a linear path), combined values flow naturally across blocks via SSA.
	var combinedMask Value
	for i, block := range chain.Blocks {
		vif := block.Instrs[len(block.Instrs)-1].(*If)
		maskCond := spmdInsertConvertToMask(block, vif.Cond)

		if i == 0 {
			combinedMask = maskCond
		} else if chain.Op == token.LAND {
			combinedMask = spmdInsertMaskAnd(block, combinedMask, maskCond)
		} else { // LOR
			combinedMask = spmdInsertMaskOr(block, combinedMask, maskCond)
		}

		// Step 2: Replace each chain block's If with a Jump to either the
		// next chain block or ThenBlock (for the last block). Remove the
		// short-circuit edge that bypasses the remaining conditions.
		var jumpTo, removeTo *BasicBlock
		if i < len(chain.Blocks)-1 {
			nextBlock := chain.Blocks[i+1]
			// For LAND: Succs[0]=nextBlock(keep), Succs[1]=ElseBlock(remove)
			// For LOR:  Succs[0]=ThenBlock(remove), Succs[1]=nextBlock(keep)
			if chain.Op == token.LAND {
				jumpTo = nextBlock
				removeTo = elseBlock
			} else {
				jumpTo = nextBlock
				removeTo = thenBlock
			}
		} else {
			// Last block always jumps to ThenBlock.
			jumpTo = thenBlock
			removeTo = elseBlock
		}
		spmdReplaceIfWithJump(block, jumpTo, removeTo)
	}

	// combinedMask is now the final combined condition, computed in the last
	// chain block which dominates ThenBlock.
	lastBlock := chain.Blocks[len(chain.Blocks)-1]

	if mergeBlock == nil {
		thenMask := spmdInsertMaskAnd(lastBlock, activeMask, combinedMask)
		if len(loopHeaderSnaps) > 0 || (elseBlock != nil && len(thenBlock.Succs) == 1 && len(elseBlock.Succs) == 1 &&
			thenBlock.Succs[0] == elseBlock.Succs[0]) {
			// Loop-header merge: both then and else jump to the same loop header.
			// Apply the same linearization as spmdLinearizeLoopHeaderMerge:
			// chain→thenBlock→elseBlock→loopHeader. This keeps elseBlock reachable
			// so LLVM does not eliminate it as dead code.
			loopHeader := thenBlock.Succs[0]
			elseMask := spmdInsertMaskAndNot(lastBlock, activeMask, combinedMask)
			spmdRewireThenToElse(thenBlock, loopHeader, elseBlock)
			loopHeader.removePred(thenBlock)
			spmdMaskMemOps(thenBlock, thenMask, lanes)
			spmdMaskMemOps(elseBlock, elseMask, lanes)
			if deferred != nil {
				*deferred = append(*deferred, &spmdDeferredMerge{
					thenBlock:  thenBlock,
					elseBlock:  elseBlock,
					mergeBlock: loopHeader,
					thenMask:   thenMask,
					elseMask:   elseMask,
					lanes:      lanes,
					phiSnaps:   loopHeaderSnaps,
				})
			}
		} else {
			// Complex CFG with no recognizable pattern: best-effort mask of
			// then block only. Else block may be unreachable.
			spmdMaskMemOps(thenBlock, thenMask, lanes)
		}
		return
	}

	// Step 3: Apply predication using the combined mask.
	thenMask := spmdInsertMaskAnd(lastBlock, activeMask, combinedMask)

	ifWithoutElse := elseBlock == mergeBlock
	var elseMask Value
	if !ifWithoutElse {
		elseMask = spmdInsertMaskAndNot(lastBlock, activeMask, combinedMask)
	}

	if ifWithoutElse {
		// if-without-else: chain → ThenBlock → ElseBlock (=merge).
		// Use snapshotted Phi edges to find the "else" (original) value.
		// For LAND: every chain block's false edge goes to merge, so Blocks[0] works.
		// For LOR: only the last chain block's false edge goes to merge.
		elsePred := chain.Blocks[0]
		if chain.Op == token.LOR {
			elsePred = chain.Blocks[len(chain.Blocks)-1]
		}
		spmdReplaceBooleanChainPhis(mergeBlock, thenBlock, elsePred, thenMask, lanes, phiSnapshots)
		spmdMaskMemOps(thenBlock, thenMask, lanes)
	} else {
		// if-else: chain → ThenBlock → ElseBlock → Merge.
		// Rewire ThenBlock's successor from Merge to ElseBlock.
		spmdRewireThenToElse(thenBlock, mergeBlock, elseBlock)
		spmdReplacePhisWithSelect(mergeBlock, thenBlock, elseBlock, thenMask, elseMask, lanes)
		spmdMaskMemOps(thenBlock, thenMask, lanes)
		spmdMaskMemOps(elseBlock, elseMask, lanes)
	}
}

// spmdReplaceBooleanChainPhis replaces Phis at mergeBlock using snapshotted
// edge values. Used for if-without-else boolean chains where chain blocks
// have been removed from mergeBlock.Preds before Phi replacement.
//
// thenPred is the then-block whose edge provides the "then" value.
// elsePred is a chain block whose snapshotted edge provides the "else" value.
func spmdReplaceBooleanChainPhis(mergeBlock, thenPred, elsePred *BasicBlock, thenMask Value, lanes int, snapshots []spmdPhiSnapshot) {
	for i, snap := range snapshots {
		thenVal, hasThen := snap.edgeVal[thenPred]
		elseVal, hasElse := snap.edgeVal[elsePred]
		if !hasThen || !hasElse {
			continue // defensive: Phi not driven by expected predecessors
		}

		sel := &SPMDSelect{
			Mask:  thenMask,
			X:     thenVal,
			Y:     elseVal,
			Lanes: lanes,
		}
		sel.setType(snap.phi.Type())
		sel.setBlock(mergeBlock)

		spmdAddReferrer(thenMask, sel)
		spmdAddReferrer(thenVal, sel)
		spmdAddReferrer(elseVal, sel)

		mergeBlock.Instrs[i] = sel
		replaceAll(snap.phi, sel)

		// Clean up old Phi referrers using the snapshotted edges
		// (the live Phi.Edges may have been compacted by removePred).
		for _, edge := range snap.edgeVal {
			if edge != nil {
				if refs := edge.Referrers(); refs != nil {
					*refs = removeInstr(*refs, snap.phi)
				}
			}
		}
		snap.phi.block = nil
	}
}

// spmdInsertBeforeTerminator inserts instr into block b just before the
// last instruction (the terminator). Sets instr.block = b.
func spmdInsertBeforeTerminator(b *BasicBlock, instr Instruction) {
	instr.setBlock(b)
	n := len(b.Instrs)
	if n == 0 {
		panic("spmdInsertBeforeTerminator: block has no instructions")
	}
	// Append a slot then shift the terminator right.
	b.Instrs = append(b.Instrs, nil)
	b.Instrs[n] = b.Instrs[n-1] // move terminator to end
	b.Instrs[n-1] = instr        // insert instr before terminator
}

// spmdRelocateToBlock moves all non-terminator instructions from src into dst,
// inserting them before dst's terminator. Each moved instruction's block pointer
// is updated to dst via setBlock. src's instructions are left as-is; the caller
// is responsible for subsequently clearing src.Instrs.
//
// This is used when src is being killed (made unreachable) but its instructions
// are referenced by instructions in dst (e.g., SPMDSelect operands). Moving them
// ensures DomPreorder traversal in TinyGo can find them via getValue().
//
// Precondition: src has at least one instruction (its terminator). dst has at
// least one instruction (its terminator). ifBlock dominates thenBlock so all
// operands of thenBlock's non-terminator instructions are available in ifBlock.
func spmdRelocateToBlock(src, dst *BasicBlock) {
	// Identify the slice of non-terminator instructions in src.
	// The terminator is the last instruction; skip it.
	n := len(src.Instrs)
	if n <= 1 {
		// src has only a terminator (or is empty); nothing to relocate.
		return
	}
	nonTerms := src.Instrs[:n-1]

	// Find insertion point in dst: just before dst's terminator (last instruction).
	dstN := len(dst.Instrs)
	if dstN == 0 {
		panic("spmdRelocateToBlock: dst block has no instructions")
	}

	// Grow dst.Instrs to hold the relocated instructions.
	dst.Instrs = append(dst.Instrs, make([]Instruction, len(nonTerms))...)
	// Shift dst's terminator to the end to make room.
	copy(dst.Instrs[dstN+len(nonTerms)-1:], dst.Instrs[dstN-1:dstN])
	// Copy non-terminator instructions from src into the gap.
	copy(dst.Instrs[dstN-1:], nonTerms)

	// Update each relocated instruction's block pointer.
	for _, instr := range nonTerms {
		instr.setBlock(dst)
	}
}

// spmdAddReferrer registers instr as a referrer of val, if val has a referrer list.
// This is needed when inserting new instructions after buildReferrers has already run.
func spmdAddReferrer(val Value, instr Instruction) {
	if refs := val.Referrers(); refs != nil {
		*refs = append(*refs, instr)
	}
}

// spmdReplaceIfWithJump replaces the If terminator of block b (which was
// "if cond goto then else else_") with a Jump to thenBlock.
// Rewires the CFG: removes b→elseBlock edge and keeps b→thenBlock.
// The If's Cond still has b as a referrer — we must clean that up.
// Sets oldIf.block = nil to mark it as removed.
func spmdReplaceIfWithJump(b *BasicBlock, thenBlock, elseBlock *BasicBlock) {
	// The old If instruction.
	oldIf := b.Instrs[len(b.Instrs)-1].(*If)

	// Remove If's Cond from referrers — the If is being deleted.
	if refs := oldIf.Cond.Referrers(); refs != nil {
		*refs = removeInstr(*refs, oldIf)
	}

	// Mark the If as removed from its block.
	oldIf.block = nil

	// Replace If with Jump.
	jmp := &Jump{}
	jmp.setBlock(b)
	b.Instrs[len(b.Instrs)-1] = jmp

	// Update CFG: b now has only thenBlock as successor.
	b.Succs = []*BasicBlock{thenBlock}

	// Remove b from elseBlock's predecessors.
	elseBlock.removePred(b)
}

// spmdRewireThenToElse rewires thenBlock so that instead of jumping to mergeBlock
// it jumps to elseBlock. Updates Preds/Succs on all affected blocks.
func spmdRewireThenToElse(thenBlock, mergeBlock, elseBlock *BasicBlock) {
	// thenBlock currently ends with Jump to mergeBlock.
	// Change its Succs[0] from mergeBlock to elseBlock.
	thenBlock.replaceSucc(mergeBlock, elseBlock)

	// Update the Jump terminator — the Jump struct itself has no target field;
	// the target is implied by Succs[0]. No instruction field update needed.

	// mergeBlock no longer has thenBlock as a predecessor.
	// We do NOT call mergeBlock.removePred(thenBlock) yet because
	// spmdReplacePhisWithSelect needs to read the Phi edges first.
	// The caller (predicateVaryingIf) calls spmdReplacePhisWithSelect
	// before this function returns, so order matters.

	// elseBlock gains thenBlock as a predecessor.
	// Invariant: elseBlock had exactly one predecessor (ifBlock, via the
	// original If branch) before linearization, so prepending is safe.
	// findMergeBlock only accepts simple diamond/fallthrough patterns where
	// this holds.
	elseBlock.Preds = append([]*BasicBlock{thenBlock}, elseBlock.Preds...)
}

// spmdReplacePhisWithSelect replaces Phi instructions at mergeBlock with SPMDSelect.
//
// Parameters:
//   - mergeBlock: the merge block containing Phi instructions to replace.
//   - thenPred:   the predecessor on the "then" side (value for active lanes).
//   - elsePred:   the predecessor on the "else" side (value for inactive lanes).
//   - thenMask:   the mask for active-lane values.
//   - elseMask:   the mask for inactive-lane values (may equal thenMask for if-without-else).
//   - lanes:      lane count for SPMDSelect.
func spmdReplacePhisWithSelect(mergeBlock, thenPred, elsePred *BasicBlock, thenMask, elseMask Value, lanes int) {
	for i := 0; i < len(mergeBlock.Instrs); i++ {
		phi, ok := mergeBlock.Instrs[i].(*Phi)
		if !ok {
			break // phis are always first
		}

		// Find the edge values from thenPred and elsePred.
		var thenVal, elseVal Value
		for j, pred := range mergeBlock.Preds {
			if pred == thenPred {
				thenVal = phi.Edges[j]
			} else if pred == elsePred {
				elseVal = phi.Edges[j]
			}
		}

		if thenVal == nil || elseVal == nil {
			// Phi is not driven by these two predecessors; leave it alone.
			// This is safe because findMergeBlock only accepts simple diamonds
			// with exactly two predecessors, so this branch is unreachable in
			// practice. It's a defensive guard for future extensions.
			continue
		}

		// Create SPMDSelect to replace this Phi.
		sel := &SPMDSelect{
			Mask:  thenMask,
			X:     thenVal,
			Y:     elseVal,
			Lanes: lanes,
		}
		sel.setType(phi.Type())
		sel.setBlock(mergeBlock)

		// Register SPMDSelect as a referrer of its operands.
		spmdAddReferrer(thenMask, sel)
		spmdAddReferrer(thenVal, sel)
		spmdAddReferrer(elseVal, sel)

		// Replace Phi with SPMDSelect in the Instrs slice.
		mergeBlock.Instrs[i] = sel

		// Update all instructions that referenced the Phi to use SPMDSelect instead.
		// replaceAll handles the referrer list update.
		replaceAll(phi, sel)

		// Clean up the Phi's operands from referrer lists.
		for _, edge := range phi.Edges {
			if edge != nil {
				if refs := edge.Referrers(); refs != nil {
					*refs = removeInstr(*refs, phi)
				}
			}
		}
		phi.block = nil
	}

	// After replacing phis, remove thenPred from mergeBlock.Preds if this
	// was an if-else linearization (thenPred no longer flows directly to mergeBlock).
	// We detect this by checking if thenPred's Succs no longer include mergeBlock.
	if !slices.Contains(thenPred.Succs, mergeBlock) {
		mergeBlock.removePred(thenPred)
	}
}


// spmdSwitchCaseInfo holds the per-case mask and body block for a linearized
// switch case. Used internally by predicateVaryingSwitch.
type spmdSwitchCaseInfo struct {
	mask      Value       // per-case active mask (lanes matching this case)
	bodyBlock *BasicBlock // case body block (then-branch of the original If)
}

// predicateVaryingSwitch linearizes a varying switch chain.
//
// For an N-case switch with optional default:
//
//	compBlock0: cond0 computed; If cond0 → body0, compBlock1
//	compBlock1: cond1 computed; If cond1 → body1, default/done
//	body0: ... Jump done
//	body1: ... Jump done
//	default: ... Jump done  (if present)
//	done: Phi [body0: v0, body1: v1, default/last: vN]
//
// Transforms to:
//
//	compBlock0: maskCond0=convert(cond0); mask0=AND(rem,maskCond0); rem=AND_NOT(rem,maskCond0); Jump body0
//	body0: ... [mem ops with mask0] Jump compBlock1
//	compBlock1: maskCond1=convert(cond1); mask1=AND(rem,maskCond1); rem=AND_NOT(rem,maskCond1); Jump body1
//	body1: ... [mem ops with mask1] Jump default/done
//	default: ... [mem ops with defaultMask] Jump done  (if present)
//	done: SPMDSelect(mask0, v0, SPMDSelect(mask1, v1, vDefault))
func predicateVaryingSwitch(fn *Function, lanes int, chain *SPMDSwitchChain, spmdLoopBlock, spmdBodyBlock *BasicBlock, activeMask Value) {
	if len(chain.Cases) == 0 {
		return
	}
	if chain.DoneBlock == nil {
		return
	}

	doneBlock := chain.DoneBlock

	// Phase 0: Snapshot Phi edge values at DoneBlock BEFORE any CFG rewiring.
	//
	// spmdRewireBodyToNext calls doneBlock.removePred(bodyBlock), which
	// removes both the predecessor entry and the corresponding Phi edge from
	// doneBlock's Phi instructions. We must capture the per-case Phi values
	// before that happens, otherwise the values for rewired cases are lost.
	//
	// phiEdges[bodyBlock] = map of phi → phi-edge-value-for-that-body.
	//
	// Skip snapshotting when doneBlock is a loop header: in that case all
	// Phis are loop-carried (not switch-merge), and Phase 3 will be skipped.
	type phiEdgeMap = map[*Phi]Value
	doneIsLoopBlock := doneBlock == spmdLoopBlock || doneBlock == spmdBodyBlock
	var phiEdges map[*BasicBlock]phiEdgeMap
	if !doneIsLoopBlock {
		phiEdges = make(map[*BasicBlock]phiEdgeMap, len(doneBlock.Preds))
		for i, pred := range doneBlock.Preds {
			m := make(phiEdgeMap)
			for _, instr := range doneBlock.Instrs {
				phi, ok := instr.(*Phi)
				if !ok {
					break
				}
				if i < len(phi.Edges) {
					m[phi] = phi.Edges[i]
				}
			}
			phiEdges[pred] = m
		}
	}

	// Phase 1: Compute per-case masks and linearize the comparison blocks.
	//
	// remaining tracks the mask of lanes not yet claimed by an earlier case.
	// It starts as the active mask and is narrowed by AND_NOT after each case.
	remaining := Value(activeMask)
	cases := make([]spmdSwitchCaseInfo, len(chain.Cases))

	for idx, caseIf := range chain.Cases {
		compBlock := caseIf.Block()
		bodyBlock := compBlock.Succs[0] // then-branch = case body
		nextBlock := compBlock.Succs[1] // else-branch = next comparison or default

		// Compute per-case mask: caseMask = remaining & convert(caseIf.Cond)
		maskCond := spmdInsertConvertToMask(compBlock, caseIf.Cond)
		caseMask := spmdInsertMaskAnd(compBlock, remaining, maskCond)

		// Narrow remaining: remaining = remaining &^ maskCond
		newRemaining := spmdInsertMaskAndNot(compBlock, remaining, maskCond)

		cases[idx] = spmdSwitchCaseInfo{mask: caseMask, bodyBlock: bodyBlock}

		// Replace the If with a Jump to the body block.
		// This removes compBlock from nextBlock's predecessors.
		spmdReplaceIfWithJump(compBlock, bodyBlock, nextBlock)

		// Rewire: redirect the body's jump so control flows sequentially
		// through all cases and the default before reaching done.
		//
		// NOTE: spmdRewireBodyToNext calls doneBlock.removePred(bodyBlock),
		// which also compacts doneBlock's Phi.Edges. This is safe because we
		// already snapshotted all Phi edge values in Phase 0.
		isLastCase := idx == len(chain.Cases)-1
		if !isLastCase {
			// Non-last case: rewire body → nextBlock (the next comparison block).
			spmdRewireBodyToNext(bodyBlock, doneBlock, nextBlock)
		} else if chain.DefaultBlock != nil {
			// Last case with a default: rewire body → defaultBlock.
			// nextBlock is the default block for the last case's If.
			spmdRewireBodyToNext(bodyBlock, doneBlock, chain.DefaultBlock)
		}
		// Last case without default: body already jumps to doneBlock — leave it.

		remaining = newRemaining
	}

	// remaining is now the default mask: lanes not matched by any explicit case.
	defaultMask := remaining

	// Phase 2: Mask memory operations in each case body and the default body.
	for _, ci := range cases {
		spmdMaskMemOps(ci.bodyBlock, ci.mask, lanes)
	}
	if chain.DefaultBlock != nil {
		spmdMaskMemOps(chain.DefaultBlock, defaultMask, lanes)
	}

	// Phase 3: Replace switch-merge Phis at DoneBlock with a chained SPMDSelect.
	//
	// When doneBlock is the loop's loop/body block (no explicit switch.done merge
	// block; case bodies jump directly back to the loop header), the switch
	// produces no live scalar values. All Phis in the loop header are loop-carried
	// — do not replace them.
	if !doneIsLoopBlock {
		spmdReplaceSwitchPhisWithChainedSelect(chain, cases, phiEdges, lanes)
	}
}

// spmdRewireBodyToNext redirects the body block's terminator from oldDest to newDest.
// The body block ends with a Jump; we update its single successor.
// oldDest loses body as a predecessor; newDest gains body as a predecessor.
func spmdRewireBodyToNext(bodyBlock, oldDest, newDest *BasicBlock) {
	bodyBlock.replaceSucc(oldDest, newDest)
	oldDest.removePred(bodyBlock)
	// Insert bodyBlock at the front of newDest.Preds so the ordering reflects
	// CFG traversal order (the comparison block was already removed from
	// newDest.Preds by spmdReplaceIfWithJump; bodyBlock now takes its place).
	// Guard against duplicate predecessors if bodyBlock already reaches newDest.
	if !slices.Contains(newDest.Preds, bodyBlock) {
		newDest.Preds = append([]*BasicBlock{bodyBlock}, newDest.Preds...)
	}
}

// spmdReplaceSwitchPhisWithChainedSelect replaces each Phi at chain.DoneBlock
// with a chained SPMDSelect.
//
// phiEdges is a snapshotted map of predecessor block → (phi → edge value),
// captured before CFG rewiring removed edges from the Phi slices.
//
// For a 2-case switch with default, the Phi [body0: v0, body1: v1, default: v2]
// becomes:
//
//	sel1 = SPMDSelect(case1Mask, v1, v2)   // case 1 vs default
//	sel0 = SPMDSelect(case0Mask, v0, sel1) // case 0 vs rest
//
// If there is no default, the last case's value is used as the base "else" value.
func spmdReplaceSwitchPhisWithChainedSelect(
	chain *SPMDSwitchChain,
	cases []spmdSwitchCaseInfo,
	phiEdges map[*BasicBlock]map[*Phi]Value,
	lanes int,
) {
	doneBlock := chain.DoneBlock

	// Snapshot the Phi instructions before we mutate the Instrs slice.
	var phis []*Phi
	for _, instr := range doneBlock.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break // phis are always first in a block
		}
		phis = append(phis, phi)
	}
	if len(phis) == 0 {
		// No phis at the done block — nothing to merge, nothing to do.
		// Phase 1 already cleaned up stale predecessor edges via
		// spmdRewireBodyToNext (which calls doneBlock.removePred for each
		// rewired case body).
		return
	}

	// The base "else" value comes from the default block (if present) or the
	// last case body. Both still flow into doneBlock after linearization.
	var baseBlock *BasicBlock
	if chain.DefaultBlock != nil {
		baseBlock = chain.DefaultBlock
	} else {
		baseBlock = cases[len(cases)-1].bodyBlock
	}

	// Collect all new instructions to insert, per-phi. We process
	// phis in order and track {instructions, outermost value} for each.
	type phiReplacement struct {
		instrs []Instruction // all SPMDSelects in dependency order (inner first)
		val    Value         // outermost value (may be a pre-existing Value, not Instruction)
		phi    *Phi
	}
	var replacements []phiReplacement

	for _, phi := range phis {
		// Look up the base value from the snapshotted edge map.
		baseEdges, ok := phiEdges[baseBlock]
		if !ok {
			continue
		}
		baseVal, ok := baseEdges[phi]
		if !ok {
			continue
		}

		// Build chained selects from last case to first (bottom-up).
		// Cases whose bodyBlock == baseBlock provide the starting value and
		// skip select emission (no-default: last case IS the base).
		var chainInstrs []Instruction
		sel := Value(baseVal)
		for i := len(cases) - 1; i >= 0; i-- {
			ci := cases[i]
			if ci.bodyBlock == baseBlock {
				continue
			}
			caseEdges, ok := phiEdges[ci.bodyBlock]
			if !ok {
				continue
			}
			caseVal, ok := caseEdges[phi]
			if !ok {
				continue
			}

			newSel := &SPMDSelect{
				Mask:  ci.mask,
				X:     caseVal,
				Y:     sel,
				Lanes: lanes,
			}
			newSel.setType(phi.Type())
			newSel.setBlock(doneBlock)
			spmdAddReferrer(ci.mask, newSel)
			spmdAddReferrer(caseVal, newSel)
			spmdAddReferrer(sel, newSel)
			chainInstrs = append(chainInstrs, newSel)
			sel = newSel
		}

		replacements = append(replacements, phiReplacement{
			instrs: chainInstrs,
			val:    sel,
			phi:    phi,
		})
	}

	// Rebuild the Instrs slice: replace each Phi with its full chain of
	// SPMDSelects (innermost first so dependencies are satisfied), then
	// append remaining non-Phi instructions.
	var newInstrs []Instruction
	phiSet := make(map[*Phi]bool, len(phis))
	for _, phi := range phis {
		phiSet[phi] = true
	}
	repIdx := 0
	for _, instr := range doneBlock.Instrs {
		if phi, ok := instr.(*Phi); ok && phiSet[phi] {
			if repIdx < len(replacements) && replacements[repIdx].phi == phi {
				r := replacements[repIdx]
				newInstrs = append(newInstrs, r.instrs...)
				repIdx++
			}
			// else: phi had no replacement (defensive skip above)
		} else {
			newInstrs = append(newInstrs, instr)
		}
	}
	doneBlock.Instrs = newInstrs

	// Redirect uses and clean up old Phis.
	for _, r := range replacements {
		replaceAll(r.phi, r.val)
		for _, edge := range r.phi.Edges {
			if edge != nil {
				if refs := edge.Referrers(); refs != nil {
					*refs = removeInstr(*refs, r.phi)
				}
			}
		}
		r.phi.block = nil
	}

	// Compact nil slots from degenerate Phi replacements.
	j := 0
	for _, instr := range doneBlock.Instrs {
		if instr != nil {
			doneBlock.Instrs[j] = instr
			j++
		}
	}
	for k := j; k < len(doneBlock.Instrs); k++ {
		doneBlock.Instrs[k] = nil // clear for GC
	}
	doneBlock.Instrs = doneBlock.Instrs[:j]

	// Predecessor cleanup was already performed in Phase 1: spmdRewireBodyToNext
	// called doneBlock.removePred(bodyBlock) for each non-last case body that was
	// rewired away from doneBlock. No further cleanup is needed here.
}

// spmdConvertAllMemOps converts remaining UnOp{MUL}/Store in fn to
// SPMDLoad/SPMDStore with the given mask. Called AFTER predicateSPMDScope
// (which already converted loads/stores inside varying branches).
// Instructions already converted (SPMDLoad/SPMDStore) are skipped by the type switch.
// Only converts loads/stores of SPMD-compatible element types (basic types,
// SPMDType). Struct/interface/map/slice stores (e.g., from fmt.Printf)
// are left as regular instructions since TinyGo can't vectorize them.
//
// No contiguity detection is performed here. spmdIsContiguousIndex checks
// against SPMDLoop IterPhis, and function bodies have no SPMDLoops.
// Contiguous access in function bodies is detected by TinyGo's
// spmdContiguousPtr map during IndexAddr compilation.
func spmdConvertAllMemOps(fn *Function, mask Value, lanes int) {
	for _, block := range fn.Blocks {
		for i := 0; i < len(block.Instrs); i++ {
			instr := block.Instrs[i]

			switch instr := instr.(type) {
			case *UnOp:
				if instr.Op != token.MUL {
					continue
				}
				// Only convert loads of SPMD-compatible element types.
				if !spmdIsVectorizableElemType(instr.Type()) {
					continue
				}
				// Replace pointer load with SPMDLoad.
				load := &SPMDLoad{
					Addr:  instr.X,
					Mask:  mask,
					Lanes: lanes,
					pos:   instr.Pos(),
				}
				load.setType(instr.Type())
				load.setBlock(block)

				// Update referrers.
				spmdAddReferrer(instr.X, load)
				spmdAddReferrer(mask, load)

				block.Instrs[i] = load

				// Replace all uses of the old UnOp with SPMDLoad.
				replaceAll(instr, load)

				// Remove old UnOp's operand referrers.
				if refs := instr.X.Referrers(); refs != nil {
					*refs = removeInstr(*refs, instr)
				}
				instr.block = nil

			case *Store:
				// Only convert stores of SPMD-compatible element types.
				if !spmdIsVectorizableElemType(instr.Val.Type()) {
					continue
				}
				// Replace store with SPMDStore.
				store := &SPMDStore{
					Addr:  instr.Addr,
					Val:   instr.Val,
					Mask:  mask,
					Lanes: lanes,
					pos:   instr.Pos(),
				}
				store.setBlock(block)

				// Update referrers.
				spmdAddReferrer(instr.Addr, store)
				spmdAddReferrer(instr.Val, store)
				spmdAddReferrer(mask, store)

				block.Instrs[i] = store

				// Remove old Store's operand referrers.
				if refs := instr.Addr.Referrers(); refs != nil {
					*refs = removeInstr(*refs, instr)
				}
				if refs := instr.Val.Referrers(); refs != nil {
					*refs = removeInstr(*refs, instr)
				}
				instr.block = nil
			}
		}
	}
}

// spmdMaskAllIndexOps sets SPMDMask on all IndexAddr/Index instructions in fn
// that have a varying (SPMDType) index and don't already have a mask assigned.
// Called from predicateSPMDFuncBody for function body scope after
// spmdConvertAllMemOps has processed loads and stores.
func spmdMaskAllIndexOps(fn *Function, mask Value) {
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			switch v := instr.(type) {
			case *IndexAddr:
				if v.SPMDMask != nil {
					continue
				}
				if _, ok := v.Index.Type().(*types.SPMDType); ok {
					v.SPMDMask = mask
					spmdAddReferrer(mask, v)
				}
			case *Index:
				if v.SPMDMask != nil {
					continue
				}
				if _, ok := v.Index.Type().(*types.SPMDType); ok {
					v.SPMDMask = mask
					spmdAddReferrer(mask, v)
				}
			}
		}
	}
}

// spmdIsVectorizableElemType reports whether t is a type that TinyGo can
// vectorize via LLVM masked load/store intrinsics. Only basic numeric/bool
// types and SPMDType are supported. Struct, interface, slice, map, array,
// and other complex types are not vectorizable.
func spmdIsVectorizableElemType(t types.Type) bool {
	switch t := t.Underlying().(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.Bool,
			types.Int8, types.Uint8,
			types.Int16, types.Uint16,
			types.Int32, types.Uint32, types.Float32,
			types.Int64, types.Uint64, types.Float64,
			types.Int, types.Uint, types.Uintptr:
			return true
		}
	case *types.Pointer:
		return true
	}
	// SPMDType is always vectorizable.
	if _, ok := t.(*types.SPMDType); ok {
		return true
	}
	return false
}

// spmdIsContiguousIndex reports whether index traces back to an SPMD loop's
// IterPhi (or scalar+IterPhi BinOp), indicating contiguous lane addresses.
// Unwraps ChangeType/Convert chains since the type checker wraps the
// iter phi in changetype Varying[int] <- int.
// Runs during predication (before peeling), so IterPhi is valid.
// For function bodies (no SPMDLoops), always returns false.
func spmdIsContiguousIndex(fn *Function, index Value) bool {
	// Unwrap ChangeType/Convert chains to find underlying value.
	unwrap := func(v Value) Value {
		for {
			switch u := v.(type) {
			case *ChangeType:
				v = u.X
			case *Convert:
				v = u.X
			default:
				return v
			}
		}
	}
	index = unwrap(index)
	for _, loop := range fn.SPMDLoops {
		if loop.IterPhi == nil {
			continue
		}
		if index == loop.IterPhi {
			return true
		}
		if binop, ok := index.(*BinOp); ok && binop.Op == token.ADD {
			if unwrap(binop.X) == loop.IterPhi || unwrap(binop.Y) == loop.IterPhi {
				return true
			}
		}
	}
	return false
}

// spmdMaskMemOps walks block b and replaces:
//   - UnOp{Op: token.MUL} (pointer load) with SPMDLoad(addr, mask, lanes)
//   - Store{} with SPMDStore(addr, val, mask, lanes)
func spmdMaskMemOps(b *BasicBlock, mask Value, lanes int) {
	for i := 0; i < len(b.Instrs)-1; i++ { // -1: skip terminator
		instr := b.Instrs[i]

		switch instr := instr.(type) {
		case *UnOp:
			if instr.Op != token.MUL {
				continue
			}
			// Replace pointer load with SPMDLoad.
			load := &SPMDLoad{
				Addr:  instr.X,
				Mask:  mask,
				Lanes: lanes,
				pos:   instr.Pos(),
			}
			// Detect contiguous access via IndexAddr with iter-based index.
			if indexAddr, ok := instr.X.(*IndexAddr); ok {
				if spmdIsContiguousIndex(b.parent, indexAddr.Index) {
					load.Contiguous = true
					load.Source = indexAddr.X
				}
			}
			load.setType(instr.Type())
			load.setBlock(b)

			// Update referrers.
			spmdAddReferrer(instr.X, load)
			spmdAddReferrer(mask, load)
			if load.Source != nil {
				spmdAddReferrer(load.Source, load)
			}

			b.Instrs[i] = load

			// Replace all uses of the old UnOp with SPMDLoad.
			replaceAll(instr, load)

			// Remove old UnOp's operand referrers.
			if refs := instr.X.Referrers(); refs != nil {
				*refs = removeInstr(*refs, instr)
			}
			instr.block = nil

		case *Store:
			// Replace store with SPMDStore.
			store := &SPMDStore{
				Addr:  instr.Addr,
				Val:   instr.Val,
				Mask:  mask,
				Lanes: lanes,
				pos:   instr.Pos(),
			}
			// Detect contiguous access via IndexAddr with iter-based index.
			if indexAddr, ok := instr.Addr.(*IndexAddr); ok {
				if spmdIsContiguousIndex(b.parent, indexAddr.Index) {
					store.Contiguous = true
					store.Source = indexAddr.X
				}
			}
			store.setBlock(b)

			// Update referrers.
			spmdAddReferrer(instr.Addr, store)
			spmdAddReferrer(instr.Val, store)
			spmdAddReferrer(mask, store)
			if store.Source != nil {
				spmdAddReferrer(store.Source, store)
			}

			b.Instrs[i] = store

			// Remove old Store's operand referrers.
			if refs := instr.Addr.Referrers(); refs != nil {
				*refs = removeInstr(*refs, instr)
			}
			if refs := instr.Val.Referrers(); refs != nil {
				*refs = removeInstr(*refs, instr)
			}
			instr.block = nil

		case *Call:
			// Set mask on calls to SPMD functions (those with varying params).
			callee := instr.Call.StaticCallee()
			if callee != nil && hasSPMDParams(callee) {
				instr.Call.SPMDMask = mask
				spmdAddReferrer(mask, instr)
			}

		case *IndexAddr:
			if instr.SPMDMask != nil {
				continue
			}
			if _, ok := instr.Index.Type().(*types.SPMDType); ok {
				instr.SPMDMask = mask
				spmdAddReferrer(mask, instr)
			}

		case *Index:
			if instr.SPMDMask != nil {
				continue
			}
			if _, ok := instr.Index.Type().(*types.SPMDType); ok {
				instr.SPMDMask = mask
				spmdAddReferrer(mask, instr)
			}
		}
	}
}

// spmdMaskCallOps sweeps all Call instructions in fn and sets their SPMDMask
// to defaultMask for any SPMD function call that doesn't already have a mask
// (i.e., calls not inside a varying if/else/switch that already received a
// narrowed mask via spmdMaskMemOps).
func spmdMaskCallOps(fn *Function, defaultMask Value) {
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			call, ok := instr.(*Call)
			if !ok || call.Call.SPMDMask != nil {
				continue
			}
			callee := call.Call.StaticCallee()
			if callee == nil || !hasSPMDParams(callee) {
				continue
			}
			call.Call.SPMDMask = defaultMask
			spmdAddReferrer(defaultMask, call)
		}
	}
}

