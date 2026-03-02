// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "go/token"

// resolveSPMDLoops resolves phi-based fields of SPMDLoopInfo after
// optimizeBlocks and lift have run. Called from finishBody.
func resolveSPMDLoops(fn *Function) {
	for _, info := range fn.SPMDLoops {
		resolveSPMDLoop(fn, info)
	}
}

func resolveSPMDLoop(fn *Function, info *SPMDLoopInfo) {
	// Step 1: Detect block fusion.
	// After optimizeBlocks, rangeInt's body+loop blocks may be fused when
	// the body has exactly one successor (the loop block) and the loop block
	// has exactly one predecessor (the body). Check whether LoopBlock is
	// still present in fn.Blocks; if not, it was merged into BodyBlock.
	if !blockInFunction(fn, info.LoopBlock) {
		info.MergedBodyLoop = true
		info.LoopBlock = info.BodyBlock
	}

	// Step 2: Find the iter phi.
	// After lift, the iter alloc is promoted to a phi that inherits the
	// alloc's Comment (see lift.go:474: phi.Comment = alloc.Comment).
	// The phi lives in:
	//   - BodyBlock for rangeInt (preds: entry block and loop/back-edge block)
	//   - LoopBlock for rangeIndexed (preds: entry block and body block)
	phiBlock := info.BodyBlock
	if info.IsRangeIndex {
		phiBlock = info.LoopBlock
	}

	if info.iterAlloc != nil {
		iterComment := info.iterAlloc.Comment
		for _, instr := range phiBlock.Instrs {
			phi, ok := instr.(*Phi)
			if !ok {
				break // phis are always first in a block
			}
			if phi.Comment == iterComment {
				info.IterPhi = phi
				break
			}
		}
		info.iterAlloc = nil // clear after resolution; no longer needed
	}

	// Step 3: Find IncrBinOp.
	// The increment is an ADD instruction of the form: iter + 1, where the
	// iter operand derives from IterPhi (possibly via a ChangeType/UnOp).
	if info.IterPhi != nil {
		info.IncrBinOp = findIncrBinOp(info.IterPhi, phiBlock)
	}

	// Step 4: Identify accumulators.
	// Any phi in phiBlock that is not the iter phi is a loop-carried value.
	// Classify each edge as "entry" (initial value) or "back-edge" (updated
	// value) by checking whether the predecessor is outside the loop.
	for _, instr := range phiBlock.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			break
		}
		if phi == info.IterPhi {
			continue
		}
		acc := classifyAccumulator(info, phi, phiBlock)
		if acc != nil {
			info.Accumulators = append(info.Accumulators, *acc)
		}
	}
}

// blockInFunction returns true if block is still present in fn.Blocks.
// After optimizeBlocks, fused blocks are removed from the slice.
func blockInFunction(fn *Function, block *BasicBlock) bool {
	for _, b := range fn.Blocks {
		if b == block {
			return true
		}
	}
	return false
}

// findIncrBinOp searches block for the ADD instruction that increments
// the loop iterator. One operand must derive from iterPhi and the other
// must be the constant 1.
func findIncrBinOp(iterPhi *Phi, block *BasicBlock) *BinOp {
	for _, instr := range block.Instrs {
		binop, ok := instr.(*BinOp)
		if !ok {
			continue
		}
		if binop.Op != token.ADD {
			continue
		}
		if isConstOne(binop.Y) && valueDerivedFrom(binop.X, iterPhi) {
			return binop
		}
		if isConstOne(binop.X) && valueDerivedFrom(binop.Y, iterPhi) {
			return binop
		}
	}
	return nil
}

// isConstOne returns true if v is the integer constant 1.
func isConstOne(v Value) bool {
	c, ok := v.(*Const)
	if !ok {
		return false
	}
	return c.Int64() == 1
}

// valueDerivedFrom returns true if v is phi or a direct UnOp of phi.
// A UnOp wrapper can appear when the SSA builder inserts a type conversion
// on the iterator before the ADD.
func valueDerivedFrom(v Value, phi *Phi) bool {
	if v == phi {
		return true
	}
	if unop, ok := v.(*UnOp); ok {
		return unop.X == phi
	}
	return false
}

// classifyAccumulator determines the init and back-edge values for a
// non-iter phi in an SPMD loop block. Returns nil if the phi does not
// have the expected two-predecessor structure.
func classifyAccumulator(info *SPMDLoopInfo, phi *Phi, phiBlock *BasicBlock) *SPMDAccumulator {
	if len(phi.Edges) != len(phiBlock.Preds) {
		return nil
	}

	var initValue, backValue Value
	for i, pred := range phiBlock.Preds {
		if isEntryPredecessor(info, pred) {
			initValue = phi.Edges[i]
		} else {
			backValue = phi.Edges[i]
		}
	}

	if initValue == nil || backValue == nil {
		return nil
	}

	return &SPMDAccumulator{
		Phi:       phi,
		InitValue: initValue,
		BackValue: backValue,
	}
}

// isEntryPredecessor returns true if block is the entry-side predecessor
// of the phi block (i.e. not a loop back-edge predecessor).
// Loop-internal blocks are BodyBlock and LoopBlock.
func isEntryPredecessor(info *SPMDLoopInfo, block *BasicBlock) bool {
	return block != info.BodyBlock && block != info.LoopBlock
}
