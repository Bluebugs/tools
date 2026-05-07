// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// accumulateSrc is the shared SPMD-loop fixture: a function with no
// varying parameters that contains an SPMD loop using a Varying[[]int]
// accumulator (non-vectorizable, so the alloca must survive lift even
// after the v8 narrowing). buildSSAWithSPMD (defined in spmd_loop_test.go)
// marks the first RangeStmt as IsSpmd=true so fn.SPMDLoops is non-empty.
//
// NOTE: updated from Varying[int] to Varying[[]int] for v8: the test must
// assert that NON-vectorizable Varying types survive lift. Varying[int] is
// vectorizable and is intentionally lifted by the v8 guard narrowing.
const accumulateSrc = `package main

import "lanes"

func accumulate(arrays [][]int) int {
	var acc lanes.Varying[[]int]
	for i := range len(arrays) {
		_ = i
		_ = acc
	}
	return 0
}

func main() {}
`

// TestSPMDV5VaryingAllocaNotLifted verifies that a non-vectorizable
// Varying[T] alloca (here Varying[[]int]) survives the SSA lift() pass —
// its Alloc instruction remains in the function body so the SPMD predication
// pass can mutate the alloca's pointee type to a width-fixed *types.SPMDType.
//
// v8 narrows the lift guard: only non-vectorizable Varying types (slice,
// struct, array, interface) are preserved. Varying[[]int] is a slice, so
// it must survive lift under v8.
func TestSPMDV5VaryingAllocaNotLifted(t *testing.T) {
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
		t.Fatal("non-vectorizable varying alloca was lifted; expected memory-backed *ssa.Alloc with Varying[[]int] element in function body")
	}
}

// TestSPMDVaryingInt32AccumulatorLifts asserts that a Varying[int32]
// accumulator alloca is REMOVED post-lift in a function with a `go for` body.
// v6.1's blanket lift guard preserved all Varying[T] allocas (perf regression);
// v8 narrows the guard to non-vectorizable Varying types so vectorizable
// accumulators (int, float, byte, pointer) lift back into phi nodes for speed.
func TestSPMDVaryingInt32AccumulatorLifts(t *testing.T) {
	src := `package main

import (
	"lanes"
	"reduce"
)

func F(data []int32) int32 {
	var total lanes.Varying[int32] = 0
	for _, v := range data {
		total += v
	}
	return reduce.Add(total)
}

func main() {}
`
	fn := buildSPMDFunction(t, src, "F")
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			alloc, ok := instr.(*ssa.Alloc)
			if !ok {
				continue
			}
			ptr, ok := alloc.Type().(*types.Pointer)
			if !ok {
				continue
			}
			spmdT, ok := ptr.Elem().(*types.SPMDType)
			if !ok {
				continue
			}
			basic, ok := spmdT.Elem().(*types.Basic)
			if !ok {
				continue
			}
			if basic.Kind() == types.Int32 {
				t.Errorf("Varying[int32] alloca survived lift: %s in block %s",
					alloc, b.Comment)
			}
		}
	}
}

// TestSPMDVaryingSlicePreserved asserts that a Varying[[]int] alloca SURVIVES
// lift. Non-vectorizable Varying types (slice, struct, array, interface) cannot
// be expressed as LLVM vector elements, so the alloca [N x T] representation
// is required. v7 Phase 1 sizes the alloca correctly; v8 keeps the lift guard
// for these types.
func TestSPMDVaryingSlicePreserved(t *testing.T) {
	// The source explicitly declares a Varying[[]int] variable inside an SPMD
	// loop. buildSSAWithSPMD marks the first RangeStmt as SPMD, triggering the
	// predication pass. The resulting Alloc must survive lift because []int is a
	// non-vectorizable element type.
	src := `package main

import "lanes"

func F(arrays [][]int) int {
	var sub lanes.Varying[[]int]
	for i := range len(arrays) {
		sub = arrays[i]
		_ = sub
	}
	return 0
}

func main() {}
`
	fn := buildSPMDFunction(t, src, "F")
	found := false
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			alloc, ok := instr.(*ssa.Alloc)
			if !ok {
				continue
			}
			ptr, ok := alloc.Type().(*types.Pointer)
			if !ok {
				continue
			}
			spmdT, ok := ptr.Elem().(*types.SPMDType)
			if !ok {
				continue
			}
			if _, isSlice := spmdT.Elem().Underlying().(*types.Slice); isSlice {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("expected Varying[[]int] alloca to survive lift, but none found")
	}
}
