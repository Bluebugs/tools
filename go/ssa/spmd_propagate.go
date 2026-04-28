// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/token"
	"go/types"
)

// spmdPropagateBlockLaneCount annotates blocks containing varying allocas
// that are consumed by SPMD operations (SPMDLoad/SPMDStore/SPMDIndex/SPMDSelect)
// in already-annotated blocks. This handles the common pattern of an
// accumulator declared in a non-SPMD function's entry block but used
// inside a `go for` loop:
//
//	var acc lanes.Varying[float64]   // entry block, SPMDLaneCount=0
//	go for i, x := range data { ... acc += x }   // loop body, SPMDLaneCount=2
//
// Without this pass, TinyGo would size `acc`'s alloca via the element-natural
// derivation (4 lanes for float64 on WASM128), mismatching the in-loop
// store width (2 lanes). Forward propagation sets the entry block's
// SPMDLaneCount = 2 so TinyGo materializes the alloca correctly.
//
// The pass handles two consumer variants:
//
//  1. SPMDLoad/SPMDStore/SPMDIndex/SPMDSelect: the production path where
//     spmdConvertLoopOps has already promoted memory ops to SPMD form
//     (requires GOEXPERIMENT=spmd with the forked type checker).
//
//  2. Plain *ssa.Store / *ssa.UnOp{token.MUL}: the test-environment path where
//     the standard importer represents lanes.Varying[T] as a *types.Named
//     struct rather than *types.SPMDType, so spmdConvertScopedMemOps skips
//     the conversion. The surrounding block is still annotated with
//     SPMDLaneCount > 0 by spmdConvertLoopOps, which is sufficient to detect
//     SPMD usage.
//
// Limitation: an entry block with two allocas of different consumer lane
// counts gets a single annotation (first SPMD consumer wins). Documented
// in the v4 design spec §6.2.
//
// Runs AFTER spmdConvertLoopOps (so consumer SPMD ops exist and in-loop
// blocks have SPMDLaneCount annotated) and BEFORE spmdSpecializeFunctions
// (so specialization sees the propagated annotations).
func spmdPropagateBlockLaneCount(fn *Function) {
	for _, b := range fn.Blocks {
		if b.SPMDLaneCount != 0 {
			continue
		}
		// Find varying allocas in this block whose first SPMD consumer is
		// in an annotated block; adopt that consumer's lane count.
		var found int
		for _, instr := range b.Instrs {
			alloc, ok := instr.(*Alloc)
			if !ok {
				continue
			}
			ptr, ok := alloc.Type().(*types.Pointer)
			if !ok {
				continue
			}
			if !isLanesVaryingType(ptr.Elem()) {
				continue
			}
			lc := spmdAllocaConsumerLaneCount(alloc)
			if lc > 0 {
				found = lc
				break
			}
		}
		if found > 0 {
			b.SPMDLaneCount = found
		}
	}
}

// spmdAllocaConsumerLaneCount walks the alloca's referrer chain and returns
// the lane count of the first SPMD consumer found in an annotated block.
//
// SPMD consumers are:
//   - SPMDLoad, SPMDStore, SPMDIndex, SPMDSelect: produced by spmdConvertLoopOps
//     in production (GOEXPERIMENT=spmd with forked type checker).
//   - *Store, *UnOp{token.MUL}: plain memory ops in annotated blocks, which
//     occur in the test environment where lanes.Varying[T] is a raw *types.Named
//     struct and spmdConvertScopedMemOps does not promote them. The annotated
//     block still signals SPMD context.
//
// Returns 0 if no annotated SPMD consumer is found.
func spmdAllocaConsumerLaneCount(alloc *Alloc) int {
	refs := alloc.Referrers()
	if refs == nil {
		return 0
	}
	for _, ref := range *refs {
		var blk *BasicBlock
		switch op := ref.(type) {
		case *SPMDLoad:
			blk = op.Block()
		case *SPMDStore:
			blk = op.Block()
		case *SPMDIndex:
			blk = op.Block()
		case *SPMDSelect:
			blk = op.Block()
		case *Store:
			// Plain store: addr must be the alloca (not the stored value).
			if op.Addr == alloc {
				blk = op.Block()
			}
		case *UnOp:
			// Plain pointer load (*ptr): operand must be the alloca.
			if op.Op == token.MUL && op.X == alloc {
				blk = op.Block()
			}
		}
		if blk != nil && blk.SPMDLaneCount > 0 {
			return blk.SPMDLaneCount
		}
	}
	return 0
}
