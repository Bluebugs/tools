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

	// Build a set of If instructions to exclude from predication.
	// Switch chains and boolean chains use IsVarying Ifs for their own
	// structural patterns (case dispatch, short-circuit evaluation) and must
	// not be linearized by this pass.
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
			continue // skip switch-chain and boolean-chain Ifs
		}
		// linearize this varying If.
		predicateVaryingIf(fn, loop, block, vif)
	}
}

// spmdBuildExcludedIfs builds a set of If instructions that should NOT be
// processed by predicateSPMD. This includes:
//   - Ifs that are part of SPMDSwitchChains (case dispatch)
//   - Ifs that are part of SPMDBooleanChains (short-circuit &&/||)
func spmdBuildExcludedIfs(fn *Function) map[*If]bool {
	excluded := make(map[*If]bool)

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
func spmdReplaceIfWithJump(b *BasicBlock, thenBlock, elseBlock *BasicBlock) {
	// The old If instruction.
	oldIf := b.Instrs[len(b.Instrs)-1].(*If)

	// Remove If's Cond from referrers — the If is being deleted.
	if refs := oldIf.Cond.Referrers(); refs != nil {
		*refs = removeInstr(*refs, oldIf)
	}

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

