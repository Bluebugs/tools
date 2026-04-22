// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// isVaryingElem reports whether t is a lanes.Varying[T] type, matching both
// the *types.SPMDType representation (produced when GOEXPERIMENT=spmd is active
// and the forked type-checker intercepts the lanes.Varying[T] type expression)
// and the *types.Named representation (produced without GOEXPERIMENT, where
// lanes.Varying[T] is a raw generic instantiation from the standard importer).
func isVaryingElem(t types.Type) bool {
	if _, ok := t.(*types.SPMDType); ok {
		return true
	}
	if named, ok := t.(*types.Named); ok {
		obj := named.Obj()
		if obj.Name() == "Varying" && obj.Pkg() != nil && obj.Pkg().Path() == "lanes" {
			return true
		}
	}
	return false
}

// TestSPMDVaryingAllocaNotLifted verifies that a Varying[T] alloca
// survives the SSA lift() pass — its Alloc instruction remains in the
// function body so the SPMD predication pass can thread the active mask
// into its Store instructions.
//
// Without this property, lift promotes the alloca to phi-nodes whose
// edge values are computed unconditionally on inactive lanes — causing
// NaN/Inf to leak through partial-mask go-for iterations.
//
// Note: the SPMDStore conversion (Store -> SPMDStore with mask) is
// validated at the TinyGo LLVM IR layer in TestSPMDVaryingLocalMaskedInTail.
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

	// Verify the acc alloca survived lift: it must appear as *ssa.Alloc
	// with a lanes.Varying[T] element (either *types.SPMDType when
	// GOEXPERIMENT=spmd is active, or *types.Named for lanes.Varying[T]
	// when the standard importer treats it as a generic instantiation).
	var gotAlloc bool
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if ptr, ok := alloc.Type().Underlying().(*types.Pointer); ok {
					if isVaryingElem(ptr.Elem()) {
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
