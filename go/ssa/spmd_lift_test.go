// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// accumulateSrc is the shared SPMD-loop fixture: a function with no varying
// parameters that contains an SPMD loop using a Varying[int] accumulator.
// buildSSAWithSPMD (defined in spmd_loop_test.go) marks the first RangeStmt
// as IsSpmd=true so fn.SPMDLoops is non-empty and spmdConvertLoopOps runs.
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
// the SSA lift() pass — its Alloc instruction remains in the function body
// so the surrounding block's SPMDLaneCount governs its TinyGo materialization.
//
// Without this property, lift promotes the alloca to phi-nodes whose edge
// values are computed unconditionally on inactive lanes — causing NaN/Inf
// to leak through partial-mask go-for iterations.
func TestSPMDVaryingAllocaNotLifted(t *testing.T) {
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
