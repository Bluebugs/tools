// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// accumulateSrc is the shared fixture for the two varying-alloca lift tests.
// accumulate has no varying parameters (so it is NOT an SPMD function body)
// but contains an SPMD loop (go for, modelled by setting IsSpmd=true on the
// RangeStmt). buildSSAWithSPMD marks the first RangeStmt as SPMD, giving
// fn.SPMDLoops a non-empty slice so spmdConvertLoopOps runs and can annotate
// the alloca with loop.LaneCount.
const accumulateSrc = `package main

import "lanes"

func accumulate(data []int) int {
	var acc lanes.Varying[int]
	for i := range len(data) {
		_ = i
		_ = acc
	}
	return 0
}

func main() {}
`

// TestSPMDVaryingAllocaNotLifted verifies that a Varying[T] alloca survives
// the SSA lift() pass — its Alloc instruction remains in the function body so
// the SPMD predication pass can annotate it with the surrounding loop's lane
// count and TinyGo can materialize its element at the correct vector width.
//
// Without this property, lift promotes the alloca to phi-nodes whose edge
// values are computed unconditionally on inactive lanes — causing NaN/Inf to
// leak through partial-mask go-for iterations.
func TestSPMDVaryingAllocaNotLifted(t *testing.T) {
	// buildSSAWithSPMD (defined in spmd_loop_test.go) builds SSA for a
	// function with an SPMD loop (marks the first RangeStmt as IsSpmd=true).
	// acc is a Varying[int] alloca; the Alloc must survive lift so the
	// predication pass can annotate it and TinyGo can emit the correct
	// vector width. Without the fix, lift promotes the alloca to phi-nodes
	// and the Alloc disappears before predication runs.
	pkg := buildSSAWithSPMD(t, accumulateSrc)
	fn := pkg.Func("accumulate")
	if fn == nil {
		t.Fatal("accumulate function not found in SSA")
	}

	var gotAlloc bool
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if ptr, ok := alloc.Type().(*types.Pointer); ok {
					if ssa.IsLanesVaryingType(ptr.Elem()) {
						gotAlloc = true
					}
				}
			}
		}
	}
	if !gotAlloc {
		t.Fatal("varying alloca was lifted; expected memory-backed *ssa.Alloc with Varying[T] element in function body")
	}
}

// TestSPMDVaryingAllocaLaneCountSet verifies that the SPMD predication pass
// annotates a Varying[T] alloca with the SPMD loop's lane count via the
// *Alloc.SPMDLaneCount field. TinyGo reads this annotation during
// alloca-type materialization to emit the correct vector width (matching the
// surrounding loop's iteration width rather than the type's register-natural
// width).
//
// SSA places all local allocas in block 0 (the entry block), not in the loop
// body blocks. spmdConvertLoopOps therefore scans fn.Blocks (all blocks) to
// reach the entry-block alloca and stamp it with loop.LaneCount.
func TestSPMDVaryingAllocaLaneCountSet(t *testing.T) {
	pkg := buildSSAWithSPMD(t, accumulateSrc)
	fn := pkg.Func("accumulate")
	if fn == nil {
		t.Fatal("accumulate function not found in SSA")
	}

	var found bool
	var laneCount int
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			alloc, ok := instr.(*ssa.Alloc)
			if !ok {
				continue
			}
			ptr, ok := alloc.Type().(*types.Pointer)
			if !ok {
				continue
			}
			if !ssa.IsLanesVaryingType(ptr.Elem()) {
				continue
			}
			found = true
			laneCount = alloc.SPMDLaneCount
		}
	}
	if !found {
		t.Fatal("varying alloca not found in function body")
	}
	if laneCount == 0 {
		t.Fatal("alloc.SPMDLaneCount is 0; expected non-zero from spmdConvertLoopOps")
	}
}
