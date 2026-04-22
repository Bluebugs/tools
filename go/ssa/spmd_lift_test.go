// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// TestSPMDVaryingAllocaNotLifted verifies that a Varying[T] alloca
// survives the SSA lift() pass — its Alloc, Store, and Load instructions
// remain in the function body so the SPMD predication pass can thread
// the active mask into them.
//
// Without this property, lift promotes the alloca to phi-nodes whose
// edge values are computed unconditionally on inactive lanes — causing
// NaN/Inf to leak through partial-mask go-for iterations.
func TestSPMDVaryingAllocaNotLifted(t *testing.T) {
	// buildSPMDFuncBody builds SSA for a function with lanes.Varying[T]
	// parameters (an SPMD function body). The type checker allows
	// lanes.Varying[int](i) conversions inside such functions.
	//
	// acc is a Varying[int] alloca; the Store inside the loop must survive
	// lift so the predication pass can convert it to an SPMDStore with the
	// active mask. Without the fix, lift promotes the alloca to phi-nodes
	// and the Store disappears before predication runs.
	src := `package main

import "lanes"

func accumulate(v lanes.Varying[int], n int) lanes.Varying[int] {
	var acc lanes.Varying[int]
	for i := range n {
		_ = i
		acc = v
	}
	return acc
}

func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("accumulate")
	if fn == nil {
		t.Fatal("accumulate function not found in SSA")
	}

	var gotAlloc bool
	var gotSPMDStore bool
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if ptr, ok := alloc.Type().Underlying().(*types.Pointer); ok {
					if _, ok := ptr.Elem().(*types.SPMDType); ok {
						gotAlloc = true
					}
				}
			}
			if _, ok := instr.(*ssa.SPMDStore); ok {
				gotSPMDStore = true
			}
		}
	}
	if !gotAlloc {
		t.Fatal("varying alloca was lifted; expected memory-backed *ssa.Alloc with SPMDType element in function body")
	}
	if !gotSPMDStore {
		t.Fatal("no *ssa.SPMDStore found; predication pass didn't see a surviving Store")
	}
}
