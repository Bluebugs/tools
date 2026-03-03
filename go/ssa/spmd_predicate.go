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
	"slices"

	spmdpkg "golang.org/x/tools/go/types/spmd"
)

// predicateSPMD transforms varying control flow in SPMD functions.
// Called from finishBody after resolveSPMDLoops.
func predicateSPMD(fn *Function) {
	if len(fn.SPMDLoops) == 0 {
		return
	}

	for _, loop := range fn.SPMDLoops {
		predicateSPMDLoop(fn, loop)
	}
}

// predicateSPMDLoop transforms varying control flow within a single SPMD loop.
func predicateSPMDLoop(fn *Function, loop *SPMDLoopInfo) {
	// Build the set of blocks that are in this loop's scope (body + loop).
	// A block is "in scope" if it is reachable from the body block and
	// dominated by the body block (i.e., it is between the loop entry and exit).
	scopeBlocks := spmdLoopScopeBlocks(loop)

	// Build a set of If instructions to exclude from the generic If predication
	// pass. Both boolean-chain and switch-chain Ifs are excluded so the generic
	// loop below does not linearize them independently. Switch-chain Ifs are
	// handled by predicateVaryingSwitch below, which processes the whole chain
	// as a unit. Boolean chains will be handled by a future phase.
	excludedIfs := spmdBuildExcludedIfs(fn)

	// Find all varying If instructions in scope and linearize them.
	// We iterate fn.Blocks in program order, which is sufficient for
	// non-nested varying Ifs. For nested mask threading (Phase 8),
	// dominator-tree order will be needed.
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
		predicateVaryingIf(fn, loop, block, vif)
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
		predicateVaryingSwitch(fn, loop, chain)
	}
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

// predicateVaryingIf linearizes a single varying If instruction.
//
// Pattern:
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
func predicateVaryingIf(fn *Function, loop *SPMDLoopInfo, ifBlock *BasicBlock, vif *If) {
	lanes := loop.LaneCount
	thenBlock := ifBlock.Succs[0] // If.Cond true → Succs[0]
	elseBlock := ifBlock.Succs[1] // If.Cond false → Succs[1]

	// Find the merge (done) block — the block both paths converge to.
	// Returns nil for complex patterns (compound booleans, multi-block chains).
	mergeBlock := findMergeBlock(thenBlock, elseBlock)
	if mergeBlock == nil {
		// Cannot safely linearize this varying If (complex CFG structure).
		// Leave it as-is; the TinyGo backend will handle it with its own masking.
		return
	}

	// Compute the active mask. For now we use the all-ones constant mask
	// (all lanes active), as we do not yet thread masks through nested ifs.
	activeMask := spmdAllOnesMask()

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
func predicateVaryingSwitch(fn *Function, loop *SPMDLoopInfo, chain *SPMDSwitchChain) {
	if len(chain.Cases) == 0 {
		return
	}
	if chain.DoneBlock == nil {
		return
	}

	lanes := loop.LaneCount
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
	doneIsLoopBlock := doneBlock == loop.LoopBlock || doneBlock == loop.BodyBlock
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

	activeMask := spmdAllOnesMask()

	// Phase 1: Compute per-case masks and linearize the comparison blocks.
	//
	// remaining tracks the mask of lanes not yet claimed by an earlier case.
	// It starts as the all-ones active mask and is narrowed by AND_NOT after
	// each case.
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

	for phiIdx, phi := range phis {
		// Look up the base value from the snapshotted edge map.
		baseEdges, ok := phiEdges[baseBlock]
		if !ok {
			// baseBlock not found in snapshot — defensive skip.
			continue
		}
		baseVal, ok := baseEdges[phi]
		if !ok {
			continue
		}

		// Build chained selects from last case to first (bottom-up).
		// Cases whose bodyBlock == baseBlock provide the starting value and
		// skip select emission (no-default: last case IS the base).
		sel := Value(baseVal)
		for i := len(cases) - 1; i >= 0; i-- {
			ci := cases[i]
			if ci.bodyBlock == baseBlock {
				// Last case in a no-default switch: already the base value.
				continue
			}
			caseEdges, ok := phiEdges[ci.bodyBlock]
			if !ok {
				continue // body block not in snapshot (defensive)
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
			sel = newSel
		}

		// Replace the Phi slot with the outermost SPMDSelect in the Instrs slice.
		if selInstr, ok := sel.(Instruction); ok {
			doneBlock.Instrs[phiIdx] = selInstr
		} else {
			// Degenerate case: single-case no-default switch where no SPMDSelect
			// was built (sel is a pre-existing Value, not an Instruction).
			// Mark the slot for removal to avoid leaving a zombie Phi with
			// block=nil in a live block's Instrs slice.
			doneBlock.Instrs[phiIdx] = nil
		}

		// Redirect all uses of the old Phi to the new value.
		replaceAll(phi, sel)

		// Clean up the Phi's remaining edge referrers (post-CFG-rewiring,
		// some edges may already have been removed by removePred above).
		for _, edge := range phi.Edges {
			if edge != nil {
				if refs := edge.Referrers(); refs != nil {
					*refs = removeInstr(*refs, phi)
				}
			}
		}
		phi.block = nil
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
			load.setType(instr.Type())
			load.setBlock(b)

			// Update referrers.
			spmdAddReferrer(instr.X, load)
			spmdAddReferrer(mask, load)

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
			store.setBlock(b)

			// Update referrers.
			spmdAddReferrer(instr.Addr, store)
			spmdAddReferrer(instr.Val, store)
			spmdAddReferrer(mask, store)

			b.Instrs[i] = store

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

