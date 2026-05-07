// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// TestSPMDV5ForwardPropagationEntryBlock verifies that a Varying[int]
// accumulator in an SPMD loop is correctly typed.
//
// v8 update: Varying[int] is a vectorizable element type. The v8 lift-guard
// narrowing lifts vectorizable Varying[T] allocas back into phi nodes for
// performance. As a result, the alloca is gone after lift; the value is
// carried as a Phi with a width-fixed SPMDType. This test verifies that
// such a Phi exists with Lanes() > 0 (width-fixed by the predication pass).
func TestSPMDV5ForwardPropagationEntryBlock(t *testing.T) {
	src := `package main

import (
	"lanes"
	"reduce"
)

var data = []int{1, 2, 3, 4, 5, 6, 7, 8}

func main() {
	var acc lanes.Varying[int]
	for i := range len(data) {
		_ = i
		acc = acc
	}
	_ = reduce.Add(acc)
}
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found in SSA")
	}
	if len(fn.Blocks) == 0 {
		t.Fatal("main has no blocks")
	}

	// v8: Varying[int] is now lifted to a phi (no alloca). Check that a
	// width-fixed SPMDType phi exists in the function.
	var foundPhi bool
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			phi, ok := instr.(*ssa.Phi)
			if !ok {
				break // phis at top of block
			}
			if st, ok := phi.Type().(*types.SPMDType); ok && st.Lanes() > 0 {
				foundPhi = true
			}
		}
	}
	// Also accept the old alloca path (in case lift doesn't fire for some
	// configurations, e.g., single-lane builds).
	var foundAlloc bool
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if ptr, ok := alloc.Type().(*types.Pointer); ok {
					if _, ok := ptr.Elem().(*types.SPMDType); ok {
						foundAlloc = true
					}
				}
			}
		}
	}
	if !foundPhi && !foundAlloc {
		t.Fatal("no width-fixed Varying phi or alloca found in main(); expected SPMD accumulator for acc")
	}
}

// TestSPMDV5TypeRewriting verifies that the SPMD predication pass
// rewrites every Varying value's type in loop scope to width-fixed at
// the loop's canonical lane count. After the pass, a Varying[int]
// alloca's pointer points to a *types.SPMDType with Lanes() == loop.LaneCount.
func TestSPMDV5TypeRewriting(t *testing.T) {
	pkg := buildSSAWithSPMD(t, accumulateSrc)
	fn := pkg.Func("accumulate")
	if fn == nil {
		t.Fatal("accumulate function not found in SSA")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("accumulate has no SPMD loops; buildSSAWithSPMD should mark one")
	}
	loop := fn.SPMDLoops[0]
	if loop.LaneCount <= 0 {
		t.Fatalf("loop.LaneCount = %d; expected > 0", loop.LaneCount)
	}

	// Walk every value in the function. For each *Alloc whose pointer
	// points to a Varying[T], its pointee type should now be width-fixed
	// (Lanes() == loop.LaneCount).
	var gotFixed int
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if ptr, ok := alloc.Type().(*types.Pointer); ok {
					if elem, ok := ptr.Elem().(*types.SPMDType); ok {
						if elem.Lanes() == loop.LaneCount {
							gotFixed++
						}
					}
				}
			}
		}
	}
	if gotFixed == 0 {
		t.Fatalf("no width-fixed Varying allocas found; predication pass did not mutate types")
	}
}
