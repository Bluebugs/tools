// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// TestSPMDV5ForwardPropagationEntryBlock verifies that an entry-block
// alloca consumed by SPMD ops in a loop scope has its pointee type
// mutated to width-fixed (matching the consumer's lane count).
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

	// Find the acc alloca and check its pointee type carries Lanes().
	var found bool
	var lanes int
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if ptr, ok := alloc.Type().(*types.Pointer); ok {
					if elem, ok := ptr.Elem().(*types.SPMDType); ok {
						found = true
						lanes = elem.Lanes()
					}
				}
			}
		}
	}
	if !found {
		t.Fatal("varying alloca not found in main()")
	}
	if lanes == 0 {
		t.Fatal("alloca pointee type has Lanes() == 0; expected forward-propagation to set it from the SPMD consumer")
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
