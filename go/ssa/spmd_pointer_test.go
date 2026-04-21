// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
)

// TestSPMDPointerVaryingFieldAccess verifies that a field access through a
// *Varying[Struct] pointer produces Varying[fieldType] rather than bare
// fieldType in the SSA FieldAddr and UnOp (load) instructions.
//
// For a receiver pointPtr of type *Varying[Point], the expression pointPtr.X
// must yield a value of type Varying[int], not int. This requires:
//
//  1. FieldAddr.Type() == *Varying[int]  (not *int)
//  2. The load of that address has type Varying[int]  (not int)
func TestSPMDPointerVaryingFieldAccess(t *testing.T) {
	// The function accesses .X and .Y through *Varying[Point].
	// We use a go-for loop so that the SPMD context is active and the
	// type-checker has already resolved pointPtr as *Varying[Point].
	// buildSSAWithSPMD marks the first RangeStmt as SPMD with LaneCount=4.
	src := `package main

import "lanes"

type Point struct{ X, Y int }

func accessFields(pointPtr *lanes.Varying[Point]) lanes.Varying[int] {
	return pointPtr.X
}

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)

	fn := pkg.Func("accessFields")
	if fn == nil {
		t.Fatal("accessFields function not found in SSA")
	}

	// Walk all instructions and look for:
	//   - A FieldAddr whose result type is *Varying[int]
	//   - A UnOp (load, token.MUL) whose result type is Varying[int]
	var foundFieldAddr bool
	var foundLoad bool

	wantVaryingInt := types.NewVarying(types.Typ[types.Int])

	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			switch v := instr.(type) {
			case *ssa.FieldAddr:
				// The FieldAddr for pointPtr.X must have type *Varying[int].
				ptrType, ok := v.Type().(*types.Pointer)
				if !ok {
					t.Errorf("FieldAddr.Type() = %v, want *types.Pointer", v.Type())
					continue
				}
				if !types.Identical(ptrType.Elem(), wantVaryingInt) {
					t.Errorf("FieldAddr.Type().Elem() = %v, want %v", ptrType.Elem(), wantVaryingInt)
					continue
				}
				foundFieldAddr = true

			case *ssa.UnOp:
				if v.Op != token.MUL {
					continue
				}
				// A load from *Varying[int] must produce Varying[int].
				addrType, ok := v.X.Type().(*types.Pointer)
				if !ok {
					continue
				}
				if !types.Identical(addrType.Elem(), wantVaryingInt) {
					continue
				}
				// This is our load. Verify the result type.
				if !types.Identical(v.Type(), wantVaryingInt) {
					t.Errorf("UnOp(MUL).Type() = %v, want %v", v.Type(), wantVaryingInt)
				}
				foundLoad = true
			}
		}
	}

	if !foundFieldAddr {
		t.Error("no FieldAddr with type *Varying[int] found; expected one for pointPtr.X")
	}
	if !foundLoad {
		t.Error("no load (UnOp MUL) with result type Varying[int] found; expected one for pointPtr.X")
	}
}

// TestSPMDVaryingPointerFieldAccess verifies that a field access through a
// Varying[*Struct] value (a per-lane pointer vector) produces:
//
//  1. FieldAddr.Type() == Varying[*fieldType]  (vector of per-lane field addresses)
//  2. UnOp(MUL).Type() == Varying[fieldType]   (gather-load of the field value)
//
// This is the mirror of TestSPMDPointerVaryingFieldAccess for the other
// direction of pointer-varying (varying pointer rather than pointer to
// varying).
func TestSPMDVaryingPointerFieldAccess(t *testing.T) {
	src := `package main

import "lanes"

type Point struct{ X, Y int }

func accessFields(pointPtrs lanes.Varying[*Point]) lanes.Varying[int] {
	return pointPtrs.X
}

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)

	fn := pkg.Func("accessFields")
	if fn == nil {
		t.Fatal("accessFields function not found in SSA")
	}

	var faddr *ssa.FieldAddr
	var load *ssa.UnOp
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			switch v := instr.(type) {
			case *ssa.FieldAddr:
				faddr = v
			case *ssa.UnOp:
				if v.Op == token.MUL {
					load = v
				}
			}
		}
	}
	if faddr == nil {
		t.Fatal("no FieldAddr found in accessFields")
	}
	if load == nil {
		t.Fatal("no UnOp(MUL) load found in accessFields")
	}

	// FieldAddr on Varying[*Point] must produce Varying[*int].
	sv, ok := faddr.Type().(*types.SPMDType)
	if !ok {
		t.Fatalf("FieldAddr.Type() = %s; want Varying[*int]", faddr.Type())
	}
	if _, ok := sv.Elem().(*types.Pointer); !ok {
		t.Fatalf("FieldAddr.Type() inner = %s; want *int", sv.Elem())
	}

	// The load must produce Varying[int].
	lsv, ok := load.Type().(*types.SPMDType)
	if !ok {
		t.Fatalf("UnOp(MUL).Type() = %s; want Varying[int]", load.Type())
	}
	if basic, ok := lsv.Elem().(*types.Basic); !ok || basic.Kind() != types.Int {
		t.Fatalf("UnOp(MUL).Type() inner = %s; want int", lsv.Elem())
	}
}

// TestSPMDVaryingPointerFieldStore verifies that a field assignment through a
// Varying[*Struct] value produces a Store with:
//
//   - Addr operand Type() == Varying[*fieldType]  (per-lane addresses)
//   - Val  operand Type() == Varying[fieldType]   (per-lane values)
//
// The backend is expected to lower this Store to a masked scatter.
func TestSPMDVaryingPointerFieldStore(t *testing.T) {
	src := `package main

import "lanes"

type Point struct{ X, Y int }

func writeFields(pointPtrs lanes.Varying[*Point], v lanes.Varying[int]) {
	pointPtrs.Y = v
}

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)

	fn := pkg.Func("writeFields")
	if fn == nil {
		t.Fatal("writeFields function not found in SSA")
	}

	var store *ssa.Store
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if s, ok := instr.(*ssa.Store); ok {
				store = s
			}
		}
	}
	if store == nil {
		t.Fatal("no Store found in writeFields")
	}

	// Addr must be Varying[*int].
	addrSv, ok := store.Addr.Type().(*types.SPMDType)
	if !ok {
		t.Fatalf("Store.Addr.Type() = %s; want Varying[*int]", store.Addr.Type())
	}
	if _, ok := addrSv.Elem().(*types.Pointer); !ok {
		t.Fatalf("Store.Addr.Type() inner = %s; want *int", addrSv.Elem())
	}

	// Val must be Varying[int].
	valSv, ok := store.Val.Type().(*types.SPMDType)
	if !ok {
		t.Fatalf("Store.Val.Type() = %s; want Varying[int]", store.Val.Type())
	}
	if basic, ok := valSv.Elem().(*types.Basic); !ok || basic.Kind() != types.Int {
		t.Fatalf("Store.Val.Type() inner = %s; want int", valSv.Elem())
	}
}
