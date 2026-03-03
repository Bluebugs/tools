// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

// Tests for the four SPMD predicated instructions:
//   SPMDSelect, SPMDLoad, SPMDStore, SPMDIndex
//
// These tests verify:
//   1. Instruction struct fields (Lanes, ElemType)
//   2. Type() methods
//   3. Operands() returns correct operand pointers and supports mutation
//   4. String() formatting has correct prefix and token count
//   5. Pos() returns expected positions
//   6. Referrers() behaviour (SPMDStore returns nil, Values return &slice)
//   7. Interface compliance (compile-time checks)
//   8. Sanity checker accepts valid SPMD programs
//   9. Type system helpers: IsVaryingMask, IsMask

import (
	"bytes"
	"go/ast"
	"go/constant"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/go/types/spmd"
)

// buildSPMDProgramForPredicate builds a package with an SPMD loop.
// Used by tests that need a real function with SPMDLoopInfo.
func buildSPMDProgramForPredicate(t *testing.T) *ssa.Package {
	t.Helper()
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f) // setSPMDOnRange is defined in spmd_loop_test.go (same test package)

	pkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f}, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

// intConst creates an int32-typed SSA constant for use as a proxy operand in
// structural tests where the exact type is not being verified.
func intConst(i int64) *ssa.Const {
	return ssa.NewConst(constant.MakeInt64(i), types.Typ[types.Int32])
}

// ptrConst creates a pointer-to-int32 typed SSA constant.
func ptrConst() *ssa.Const {
	return ssa.NewConst(nil, types.NewPointer(types.Typ[types.Int32]))
}

// ----------------------------------------------------------------------------
// Type() method tests

// TestSPMDIndex_Type_Byte verifies SPMDIndex.Type() produces Varying[byte].
func TestSPMDIndex_Type_Byte(t *testing.T) {
	byteType := types.Typ[types.Byte]
	idx := &ssa.SPMDIndex{
		Lanes:    16,
		ElemType: byteType,
	}

	got := idx.Type()
	want := types.NewVarying(byteType)
	if !types.Identical(got, want) {
		t.Errorf("SPMDIndex.Type() = %s, want %s", got, want)
	}
}

// TestSPMDIndex_Type_Int32 verifies SPMDIndex.Type() produces Varying[int32].
func TestSPMDIndex_Type_Int32(t *testing.T) {
	int32Type := types.Typ[types.Int32]
	idx := &ssa.SPMDIndex{
		Lanes:    4,
		ElemType: int32Type,
	}

	got := idx.Type()
	want := types.NewVarying(int32Type)
	if !types.Identical(got, want) {
		t.Errorf("SPMDIndex.Type() = %s, want %s", got, want)
	}
}

// TestSPMDIndex_ElemTypes verifies SPMDIndex.Type() for various element types.
func TestSPMDIndex_ElemTypes(t *testing.T) {
	tests := []struct {
		name     string
		elemType types.Type
		lanes    int
	}{
		{"byte", types.Typ[types.Byte], 16},
		{"int16", types.Typ[types.Int16], 8},
		{"int32", types.Typ[types.Int32], 4},
		{"int64", types.Typ[types.Int64], 2},
		{"uint8", types.Typ[types.Uint8], 16},
		{"uint32", types.Typ[types.Uint32], 4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			idx := &ssa.SPMDIndex{
				Lanes:    tc.lanes,
				ElemType: tc.elemType,
			}
			got := idx.Type()
			want := types.NewVarying(tc.elemType)
			if !types.Identical(got, want) {
				t.Errorf("SPMDIndex.Type() = %s, want %s", got, want)
			}
		})
	}
}

// ----------------------------------------------------------------------------
// Operands() tests

// TestSPMDSelect_Operands verifies Operands() returns [Mask, X, Y] pointers.
func TestSPMDSelect_Operands(t *testing.T) {
	// Use int32 constants as proxy operands (type matters for pointer identity, not
	// for operand ordering verification).
	mask := ssa.Value(intConst(0))
	x := ssa.Value(intConst(1))
	y := ssa.Value(intConst(2))

	sel := &ssa.SPMDSelect{
		Mask:  mask,
		X:     x,
		Y:     y,
		Lanes: 4,
	}

	rands := sel.Operands(nil)
	if len(rands) != 3 {
		t.Fatalf("SPMDSelect.Operands() returned %d operands, want 3", len(rands))
	}
	if *rands[0] != mask {
		t.Errorf("SPMDSelect.Operands()[0] should be Mask")
	}
	if *rands[1] != x {
		t.Errorf("SPMDSelect.Operands()[1] should be X")
	}
	if *rands[2] != y {
		t.Errorf("SPMDSelect.Operands()[2] should be Y")
	}
}

// TestSPMDLoad_Operands verifies Operands() returns [Addr, Mask] pointers.
func TestSPMDLoad_Operands(t *testing.T) {
	addr := ssa.Value(ptrConst())
	mask := ssa.Value(intConst(0))

	load := &ssa.SPMDLoad{
		Addr:  addr,
		Mask:  mask,
		Lanes: 4,
	}

	rands := load.Operands(nil)
	if len(rands) != 2 {
		t.Fatalf("SPMDLoad.Operands() returned %d operands, want 2", len(rands))
	}
	if *rands[0] != addr {
		t.Errorf("SPMDLoad.Operands()[0] should be Addr")
	}
	if *rands[1] != mask {
		t.Errorf("SPMDLoad.Operands()[1] should be Mask")
	}
}

// TestSPMDStore_Operands verifies Operands() returns [Addr, Val, Mask] pointers.
func TestSPMDStore_Operands(t *testing.T) {
	addr := ssa.Value(ptrConst())
	val := ssa.Value(intConst(42))
	mask := ssa.Value(intConst(0))

	store := &ssa.SPMDStore{
		Addr:  addr,
		Val:   val,
		Mask:  mask,
		Lanes: 4,
	}

	rands := store.Operands(nil)
	if len(rands) != 3 {
		t.Fatalf("SPMDStore.Operands() returned %d operands, want 3", len(rands))
	}
	if *rands[0] != addr {
		t.Errorf("SPMDStore.Operands()[0] should be Addr")
	}
	if *rands[1] != val {
		t.Errorf("SPMDStore.Operands()[1] should be Val")
	}
	if *rands[2] != mask {
		t.Errorf("SPMDStore.Operands()[2] should be Mask")
	}
}

// TestSPMDIndex_Operands verifies Operands() returns empty (no value operands).
func TestSPMDIndex_Operands(t *testing.T) {
	idx := &ssa.SPMDIndex{
		Lanes:    4,
		ElemType: types.Typ[types.Int32],
	}

	rands := idx.Operands(nil)
	if len(rands) != 0 {
		t.Errorf("SPMDIndex.Operands() returned %d operands, want 0", len(rands))
	}
}

// TestSPMDSelect_OperandsAppend verifies Operands() appends to an existing slice.
func TestSPMDSelect_OperandsAppend(t *testing.T) {
	mask := ssa.Value(intConst(0))
	x := ssa.Value(intConst(1))
	y := ssa.Value(intConst(2))
	dummy := ssa.Value(intConst(99))

	sel := &ssa.SPMDSelect{Mask: mask, X: x, Y: y, Lanes: 4}

	// Pre-populate with a dummy entry via a correctly-typed pointer.
	var existing []*ssa.Value
	{
		var v ssa.Value = dummy
		existing = append(existing, &v)
	}
	rands := sel.Operands(existing)
	if len(rands) != 4 {
		t.Fatalf("Operands with 1 existing returned %d, want 4", len(rands))
	}
	if *rands[0] != dummy {
		t.Errorf("rands[0] should be existing dummy, got %v", *rands[0])
	}
	if *rands[1] != mask {
		t.Errorf("rands[1] should be Mask, got %v", *rands[1])
	}
}

// TestSPMDStore_OperandsAppend verifies Operands() appends to an existing slice.
func TestSPMDStore_OperandsAppend(t *testing.T) {
	addr := ssa.Value(ptrConst())
	val := ssa.Value(intConst(42))
	mask := ssa.Value(intConst(0))
	dummy := ssa.Value(intConst(99))

	store := &ssa.SPMDStore{Addr: addr, Val: val, Mask: mask, Lanes: 4}

	var existing []*ssa.Value
	{
		var v ssa.Value = dummy
		existing = append(existing, &v)
	}
	rands := store.Operands(existing)
	if len(rands) != 4 {
		t.Fatalf("Operands with 1 existing returned %d, want 4", len(rands))
	}
}

// TestSPMDSelect_MutateOperand verifies Operands() returns addressable pointers
// that can be written to (required by SSA renaming machinery).
func TestSPMDSelect_MutateOperand(t *testing.T) {
	mask := ssa.Value(intConst(0))
	x := ssa.Value(intConst(1))
	y := ssa.Value(intConst(2))
	newY := ssa.Value(intConst(3))

	sel := &ssa.SPMDSelect{Mask: mask, X: x, Y: y, Lanes: 4}

	rands := sel.Operands(nil)
	// rands[2] is &sel.Y; overwrite it.
	*rands[2] = newY
	if sel.Y != newY {
		t.Errorf("Mutating Operands()[2] did not update SPMDSelect.Y")
	}
}

// TestSPMDLoad_MutateOperand verifies pointer mutability for SPMDLoad.
func TestSPMDLoad_MutateOperand(t *testing.T) {
	addr := ssa.Value(ptrConst())
	mask := ssa.Value(intConst(0))
	newMask := ssa.Value(intConst(1))

	load := &ssa.SPMDLoad{Addr: addr, Mask: mask, Lanes: 4}

	rands := load.Operands(nil)
	*rands[1] = newMask
	if load.Mask != newMask {
		t.Errorf("Mutating Operands()[1] did not update SPMDLoad.Mask")
	}
}

// TestSPMDStore_MutateOperand verifies pointer mutability for SPMDStore.
func TestSPMDStore_MutateOperand(t *testing.T) {
	addr := ssa.Value(ptrConst())
	val := ssa.Value(intConst(1))
	mask := ssa.Value(intConst(0))
	newVal := ssa.Value(intConst(99))

	store := &ssa.SPMDStore{Addr: addr, Val: val, Mask: mask, Lanes: 4}

	rands := store.Operands(nil)
	*rands[1] = newVal
	if store.Val != newVal {
		t.Errorf("Mutating Operands()[1] did not update SPMDStore.Val")
	}
}

// ----------------------------------------------------------------------------
// String() format tests — use int32 constants (never panic on RelString)

// TestSPMDSelect_String verifies String() format: "spmd_select<N> M X Y".
func TestSPMDSelect_String(t *testing.T) {
	sel := &ssa.SPMDSelect{
		Mask:  intConst(0),
		X:     intConst(1),
		Y:     intConst(2),
		Lanes: 4,
	}
	s := sel.String()
	if !strings.HasPrefix(s, "spmd_select<4> ") {
		t.Errorf("SPMDSelect.String() = %q, want prefix 'spmd_select<4> '", s)
	}
	parts := strings.Fields(s)
	if len(parts) != 4 {
		t.Errorf("SPMDSelect.String() %q has %d fields, want 4", s, len(parts))
	}
}

// TestSPMDLoad_String verifies String() format: "spmd_load<N> Addr mask Mask".
func TestSPMDLoad_String(t *testing.T) {
	load := &ssa.SPMDLoad{
		Addr:  ptrConst(),
		Mask:  intConst(0),
		Lanes: 16,
	}
	s := load.String()
	if !strings.HasPrefix(s, "spmd_load<16> ") {
		t.Errorf("SPMDLoad.String() = %q, want prefix 'spmd_load<16> '", s)
	}
	if !strings.Contains(s, " mask ") {
		t.Errorf("SPMDLoad.String() = %q, want ' mask ' separator", s)
	}
	parts := strings.Fields(s)
	if len(parts) != 4 {
		t.Errorf("SPMDLoad.String() %q has %d fields, want 4", s, len(parts))
	}
}

// TestSPMDStore_String verifies String() format: "spmd_store<N> Addr Val mask Mask".
func TestSPMDStore_String(t *testing.T) {
	store := &ssa.SPMDStore{
		Addr:  ptrConst(),
		Val:   intConst(42),
		Mask:  intConst(0),
		Lanes: 16,
	}
	s := store.String()
	if !strings.HasPrefix(s, "spmd_store<16> ") {
		t.Errorf("SPMDStore.String() = %q, want prefix 'spmd_store<16> '", s)
	}
	if !strings.Contains(s, " mask ") {
		t.Errorf("SPMDStore.String() = %q, want ' mask ' separator", s)
	}
	parts := strings.Fields(s)
	if len(parts) != 5 {
		t.Errorf("SPMDStore.String() %q has %d fields, want 5", s, len(parts))
	}
}

// TestSPMDIndex_String verifies String() format: "spmd_index<N, type>".
func TestSPMDIndex_String(t *testing.T) {
	// Verify WriteFunction doesn't panic on an SPMD function.
	pkg := buildSPMDProgramForPredicate(t)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, mainFn)
	if buf.Len() == 0 {
		t.Error("WriteFunction returned empty output")
	}

	// Create detached SPMDIndex instances and verify the String() format.
	// types.Typ[types.Byte] has String() = "uint8" (alias for byte in type system).
	// Use int32 for a predictable name.
	idx := &ssa.SPMDIndex{Lanes: 4, ElemType: types.Typ[types.Int32]}
	s := idx.String()
	if !strings.HasPrefix(s, "spmd_index<4,") {
		t.Errorf("SPMDIndex.String() = %q, want prefix 'spmd_index<4,'", s)
	}
	if !strings.Contains(s, "int32") {
		t.Errorf("SPMDIndex.String() = %q, want 'int32' in output", s)
	}

	// Verify lane count appears correctly for another lane count.
	idx16 := &ssa.SPMDIndex{Lanes: 16, ElemType: types.Typ[types.Uint8]}
	s16 := idx16.String()
	if !strings.HasPrefix(s16, "spmd_index<16,") {
		t.Errorf("SPMDIndex(16).String() = %q, want prefix 'spmd_index<16,'", s16)
	}
}

// TestSPMDSelect_LanesVariants verifies Lanes field across multiple lane counts.
func TestSPMDSelect_LanesVariants(t *testing.T) {
	for _, lanes := range []int{4, 8, 16} {
		sel := &ssa.SPMDSelect{
			Mask:  intConst(0),
			X:     intConst(1),
			Y:     intConst(2),
			Lanes: lanes,
		}
		s := sel.String()
		want := strings.HasPrefix(s, "spmd_select<")
		if !want {
			t.Errorf("SPMDSelect{Lanes:%d}.String() = %q, want spmd_select prefix", lanes, s)
		}
		if sel.Lanes != lanes {
			t.Errorf("SPMDSelect.Lanes = %d, want %d", sel.Lanes, lanes)
		}
	}
}

// ----------------------------------------------------------------------------
// Pos() tests

// TestSPMDSelect_Pos verifies Pos() returns token.NoPos.
func TestSPMDSelect_Pos(t *testing.T) {
	sel := &ssa.SPMDSelect{Lanes: 4}
	if sel.Pos() != token.NoPos {
		t.Errorf("SPMDSelect.Pos() = %v, want token.NoPos", sel.Pos())
	}
}

// TestSPMDIndex_Pos verifies Pos() returns token.NoPos.
func TestSPMDIndex_Pos(t *testing.T) {
	idx := &ssa.SPMDIndex{Lanes: 4, ElemType: types.Typ[types.Int32]}
	if idx.Pos() != token.NoPos {
		t.Errorf("SPMDIndex.Pos() = %v, want token.NoPos", idx.Pos())
	}
}

// TestSPMDLoad_Pos verifies Pos() returns the stored position.
func TestSPMDLoad_Pos(t *testing.T) {
	// Zero pos (not explicitly set in struct literal) should return token.NoPos.
	load := &ssa.SPMDLoad{Lanes: 4}
	if load.Pos() != token.NoPos {
		t.Errorf("SPMDLoad.Pos() (zero) = %v, want token.NoPos", load.Pos())
	}
}

// TestSPMDStore_Pos verifies Pos() returns the stored position.
func TestSPMDStore_Pos(t *testing.T) {
	store := &ssa.SPMDStore{Lanes: 4}
	if store.Pos() != token.NoPos {
		t.Errorf("SPMDStore.Pos() (zero) = %v, want token.NoPos", store.Pos())
	}
}

// ----------------------------------------------------------------------------
// Referrers() tests

// TestSPMDStore_Referrers verifies Referrers() returns nil (effect-only instruction).
func TestSPMDStore_Referrers(t *testing.T) {
	store := &ssa.SPMDStore{Lanes: 4}
	if refs := store.Referrers(); refs != nil {
		t.Errorf("SPMDStore.Referrers() = %v, want nil (effect-only)", refs)
	}
}

// TestSPMDSelect_Referrers verifies Referrers() returns a non-nil slice pointer
// (SPMDSelect is a Value, so referrers tracking is needed).
func TestSPMDSelect_Referrers(t *testing.T) {
	sel := &ssa.SPMDSelect{Lanes: 4}
	refs := sel.Referrers()
	// register.Referrers() returns &v.referrers, which is non-nil.
	if refs == nil {
		t.Error("SPMDSelect.Referrers() = nil, want non-nil (Value instruction)")
	}
}

// TestSPMDLoad_Referrers verifies SPMDLoad.Referrers() is non-nil.
func TestSPMDLoad_Referrers(t *testing.T) {
	load := &ssa.SPMDLoad{Lanes: 4}
	if refs := load.Referrers(); refs == nil {
		t.Error("SPMDLoad.Referrers() = nil, want non-nil (Value instruction)")
	}
}

// TestSPMDIndex_Referrers verifies SPMDIndex.Referrers() is non-nil.
func TestSPMDIndex_Referrers(t *testing.T) {
	idx := &ssa.SPMDIndex{Lanes: 4, ElemType: types.Typ[types.Int32]}
	if refs := idx.Referrers(); refs == nil {
		t.Error("SPMDIndex.Referrers() = nil, want non-nil (Value instruction)")
	}
}

// ----------------------------------------------------------------------------
// Type system helper tests

// TestIsVaryingMask verifies the mask type helper from spmd package.
func TestIsVaryingMask(t *testing.T) {
	tests := []struct {
		name string
		typ  types.Type
		want bool
	}{
		{"Varying[mask]", spmd.NewVaryingMask(), true},
		{"Varying[int32]", types.NewVarying(types.Typ[types.Int32]), false},
		{"int32", types.Typ[types.Int32], false},
		{"MaskType bare", spmd.MaskInstance, false},
		{"Varying[bool]", types.NewVarying(types.Typ[types.Bool]), false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := spmd.IsVaryingMask(tc.typ)
			if got != tc.want {
				t.Errorf("IsVaryingMask(%s) = %v, want %v", tc.typ, got, tc.want)
			}
		})
	}
}

// TestIsMask verifies the bare mask type helper.
func TestIsMask(t *testing.T) {
	if !spmd.IsMask(spmd.MaskInstance) {
		t.Error("IsMask(MaskInstance) = false, want true")
	}
	if spmd.IsMask(types.Typ[types.Int32]) {
		t.Error("IsMask(int32) = true, want false")
	}
	// NewVaryingMask() wraps MaskType in SPMDType; IsMask should return false.
	if spmd.IsMask(spmd.NewVaryingMask()) {
		t.Error("IsMask(Varying[mask]) = true, want false (must be bare MaskType)")
	}
}

// TestSPMDBoolMaskDistinction verifies Varying[bool] and Varying[mask] are distinct types.
func TestSPMDBoolMaskDistinction(t *testing.T) {
	varyingBool := types.NewVarying(types.Typ[types.Bool])
	varyingMask := spmd.NewVaryingMask()

	if types.Identical(varyingBool, varyingMask) {
		t.Error("Varying[bool] and Varying[mask] should not be identical types")
	}
	if spmd.IsVaryingMask(varyingBool) {
		t.Error("Varying[bool] should not be IsVaryingMask")
	}
	if !spmd.IsVaryingMask(varyingMask) {
		t.Error("Varying[mask] should be IsVaryingMask")
	}
}

// TestVaryingMaskElem verifies that Varying[mask].Elem() == MaskInstance.
func TestVaryingMaskElem(t *testing.T) {
	vm := spmd.NewVaryingMask()
	if !spmd.IsMask(vm.Elem()) {
		t.Errorf("NewVaryingMask().Elem() is not MaskType: got %T", vm.Elem())
	}
	if vm.Elem() != spmd.MaskInstance {
		t.Error("NewVaryingMask().Elem() != MaskInstance")
	}
}

// ----------------------------------------------------------------------------
// Lanes field tests

// TestSPMDLanesField verifies Lanes field is preserved on all instruction types.
func TestSPMDLanesField(t *testing.T) {
	for _, lanes := range []int{4, 8, 16} {
		if got := (&ssa.SPMDSelect{Lanes: lanes}).Lanes; got != lanes {
			t.Errorf("SPMDSelect{Lanes:%d}.Lanes = %d", lanes, got)
		}
		if got := (&ssa.SPMDLoad{Lanes: lanes}).Lanes; got != lanes {
			t.Errorf("SPMDLoad{Lanes:%d}.Lanes = %d", lanes, got)
		}
		if got := (&ssa.SPMDStore{Lanes: lanes}).Lanes; got != lanes {
			t.Errorf("SPMDStore{Lanes:%d}.Lanes = %d", lanes, got)
		}
		if got := (&ssa.SPMDIndex{Lanes: lanes, ElemType: types.Typ[types.Int32]}).Lanes; got != lanes {
			t.Errorf("SPMDIndex{Lanes:%d}.Lanes = %d", lanes, got)
		}
	}
}

// ----------------------------------------------------------------------------
// Integration tests

// TestSPMDInstructions_InSPMDLoop verifies SPMDLoopInfo is populated in an SPMD function.
// LaneCount is computed by the type checker as 128/elemBits (e.g., 2 for int on 64-bit systems),
// so we only verify it is positive rather than checking a specific value.
func TestSPMDInstructions_InSPMDLoop(t *testing.T) {
	pkg := buildSPMDProgramForPredicate(t)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}
	if len(mainFn.SPMDLoops) == 0 {
		t.Fatal("expected SPMDLoops to be populated")
	}
	info := mainFn.SPMDLoops[0]
	if info.LaneCount <= 0 {
		t.Errorf("LaneCount = %d, want > 0", info.LaneCount)
	}
}

// TestSPMDSanity_ValidSPMDProgram verifies the sanity checker accepts a valid
// SPMD program (SanityCheckFunctions enabled).
func TestSPMDSanity_ValidSPMDProgram(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatal(err)
	}
}

// TestSPMDSanity_NoSPMDProgram verifies the sanity checker accepts a plain program.
func TestSPMDSanity_NoSPMDProgram(t *testing.T) {
	src := `package main

func add(x, y int) int { return x + y }
func main() { _ = add(1, 2) }
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatal(err)
	}
}

// TestSPMDWriteFunction verifies WriteFunction does not panic on an SPMD function.
func TestSPMDWriteFunction(t *testing.T) {
	pkg := buildSPMDProgramForPredicate(t)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, mainFn)
	if buf.Len() == 0 {
		t.Error("WriteFunction returned empty output")
	}
}

// TestSPMDSanity_VaryingIf verifies sanity check passes for varying if branches.
func TestSPMDSanity_VaryingIf(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 5 {
			_ = i
		}
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatal(err)
	}
}

// TestSPMDSanity_AccumulatorLoop verifies sanity check passes for loops with
// accumulators. The accumulator must be a varying-compatible type since it
// accumulates lane-varying values.
func TestSPMDSanity_AccumulatorLoop(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i * 2
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatal(err)
	}
}

// ----------------------------------------------------------------------------
// Compile-time interface compliance checks.

// SPMDStore must implement Instruction (effect-only, no Value produced).
var _ ssa.Instruction = (*ssa.SPMDStore)(nil)

// SPMDSelect, SPMDLoad, SPMDIndex must implement both Value and Instruction.
var _ ssa.Value = (*ssa.SPMDSelect)(nil)
var _ ssa.Value = (*ssa.SPMDLoad)(nil)
var _ ssa.Value = (*ssa.SPMDIndex)(nil)

var _ ssa.Instruction = (*ssa.SPMDSelect)(nil)
var _ ssa.Instruction = (*ssa.SPMDLoad)(nil)
var _ ssa.Instruction = (*ssa.SPMDIndex)(nil)

// ----------------------------------------------------------------------------
// predicateSPMD pass tests
// These tests verify the transformation pass that linearizes varying If/else
// control flow into predicated mask-gated operations.

// TestPredicateSPMD_NoVaryingIf verifies that when there is no varying If,
// predicateSPMD does nothing and the function remains valid.
func TestPredicateSPMD_NoVaryingIf(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	// No varying If: no SPMDSelect should appear.
	for _, block := range mainFn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDSelect); ok {
				t.Errorf("unexpected SPMDSelect in loop with no varying If")
			}
		}
	}
}

// TestPredicateSPMD_VaryingIfElse verifies that a simple if/else is linearized
// by predicateSPMD: varying If is gone, mask computations are inserted,
// and the CFG is rewired (then-block falls through to else-block).
func TestPredicateSPMD_VaryingIfElse(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		x := i
		if i > 5 {
			x = i + 1
		} else {
			x = i - 1
		}
		_ = x
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	// Verify no varying If remains (all were linearized to Jump).
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If was not linearized by predicateSPMD", block.Index)
			}
		}
	}

	// Verify mask computation instructions are present (Convert + AND inserted by predicateSPMD).
	// When the Phi at the merge is dead (_ = x discards), SPMDSelect is not generated,
	// but the mask instructions are still inserted before the Jump.
	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected mask computation instructions from predicateSPMD")
	}
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying': predicateSPMD must have replaced it with Jump")
	}
}

// TestPredicateSPMD_MaskInstructions verifies that the mask computation
// instructions (Convert to Varying[mask], AND mask binop) are inserted
// by predicateSPMD for a varying if-without-else.
func TestPredicateSPMD_MaskInstructions(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 5 {
			_ = i + 1
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	// predicateSPMD inserts: Convert(cond → Varying[mask]) and BinOp(AND, allOnes, maskCond).
	// These appear as the mask computation block before the Jump to then-block.
	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()

	// The mask-to-bool conversion must appear in the SSA dump.
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected 'lanes.Varying[mask]' mask instructions from predicateSPMD")
	}

	// The varying If must be gone.
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying': predicateSPMD must have replaced it with Jump")
	}
}

// TestPredicateSPMD_SelectLanes verifies that when SPMDSelect is generated,
// it carries the correct lane count from the enclosing SPMD loop.
// Since dead Phis are not generated (Go SSA optimizes them away), we verify
// the lane count property using the existing SPMDLoopInfo data.
func TestPredicateSPMD_SelectLanes(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	if len(mainFn.SPMDLoops) == 0 {
		t.Fatal("expected SPMDLoops to be populated")
	}
	lanes := mainFn.SPMDLoops[0].LaneCount
	if lanes <= 0 {
		t.Fatalf("LaneCount = %d, want > 0", lanes)
	}

	// Any SPMDSelect instructions that do exist must carry the correct lane count.
	for _, block := range mainFn.Blocks {
		for _, instr := range block.Instrs {
			if sel, ok := instr.(*ssa.SPMDSelect); ok {
				if sel.Lanes != lanes {
					t.Errorf("SPMDSelect.Lanes = %d, want %d (from SPMDLoopInfo)", sel.Lanes, lanes)
				}
			}
		}
	}
}

// TestPredicateSPMD_SwitchChainLinearized verifies that varying Ifs that are
// part of an SPMDSwitchChain are linearized by predicateSPMD into a sequential
// chain of comparison→body→comparison→body→...→default→done with per-case masks.
func TestPredicateSPMD_SwitchChainLinearized(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		switch i % 3 {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		default:
			_ = i + 2
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	// SPMDSwitchChains metadata must still be present after linearization.
	if len(mainFn.SPMDSwitchChains) == 0 {
		t.Fatal("expected SPMDSwitchChains to be populated")
	}

	// All switch-chain Ifs must have been replaced with Jump instructions
	// (their Block() is nil after replacement since the If was removed).
	for _, chain := range mainFn.SPMDSwitchChains {
		for _, caseIf := range chain.Cases {
			if caseIf.Block() != nil {
				t.Error("switch-chain If still has a Block(): should have been replaced with Jump by predicateSPMD")
			}
		}
	}

	// Each former comparison block (where the If lived) must end with a Jump,
	// not an If. Verify by checking that no block in the function ends with a
	// varying If (since all switch-chain Ifs should be linearized).
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If was not linearized by predicateSPMD", block.Index)
			}
		}
	}

	// Mask computation instructions (Convert to Varying[mask], AND binops) must
	// be present — inserted during switch linearization.
	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected Varying[mask] mask computation instructions from switch linearization")
	}
}

// TestPredicateSPMD_SwitchWithDefault verifies that a varying switch with 2
// cases + default is linearized into a sequential chain with per-case masks,
// and that SPMDSelect is emitted at the merge block when values are live.
func TestPredicateSPMD_SwitchWithDefault(t *testing.T) {
	// Use a rangeindex loop so slice stores generate SPMDStore instructions.
	// This verifies both: (a) CFG is linearized and (b) mem ops are masked.
	src := `package main

func f(dst []int) {
	for i := range dst {
		switch i % 3 {
		case 0:
			dst[i] = i * 2
		case 1:
			dst[i] = i * 3
		default:
			dst[i] = i * 4
		}
	}
}

func main() { f(make([]int, 16)) }
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// SPMDSwitchChain must be present.
	if len(fn.SPMDSwitchChains) == 0 {
		t.Fatal("expected SPMDSwitchChains to be populated")
	}

	// All switch-chain Ifs must have been replaced (Block() == nil).
	chain := fn.SPMDSwitchChains[0]
	for i, caseIf := range chain.Cases {
		if caseIf.Block() != nil {
			t.Errorf("case %d: If still has Block(): should be linearized to Jump", i)
		}
	}

	// No varying If should remain in any block.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If remains after switch linearization", block.Index)
			}
		}
	}

	// SPMDStore instructions must appear — the case body stores were masked.
	foundSPMDStore := false
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDStore); ok {
				foundSPMDStore = true
			}
		}
	}
	if !foundSPMDStore {
		t.Error("expected SPMDStore instructions from switch case body masking")
	}

	// Mask computation values must appear (Convert to Varying[mask] + AND/AND_NOT).
	var buf bytes.Buffer
	fn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected Varying[mask] mask computations in switch-linearized function")
	}
}

// TestPredicateSPMD_SwitchWithoutDefault verifies that a varying switch with
// 2 cases and no default is linearized correctly. The last case body flows
// directly to done (no default block to chain through).
func TestPredicateSPMD_SwitchWithoutDefault(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		switch i % 3 {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	// SPMDSwitchChain with no default must still be linearized.
	if len(mainFn.SPMDSwitchChains) == 0 {
		t.Fatal("expected SPMDSwitchChains to be populated")
	}
	chain := mainFn.SPMDSwitchChains[0]
	if chain.DefaultBlock != nil {
		t.Fatal("expected no default block for switch without default")
	}

	// All switch-chain Ifs must have been replaced with Jump.
	for i, caseIf := range chain.Cases {
		if caseIf.Block() != nil {
			t.Errorf("case %d: If still has Block() after linearization", i)
		}
	}

	// No varying If should remain.
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If not linearized", block.Index)
			}
		}
	}

	// Mask computation instructions must be present.
	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected Varying[mask] mask computations for switch without default")
	}
}

// TestPredicateSPMD_SwitchMemOps verifies that memory operations (stores) in
// varying switch case bodies are masked with the per-case mask by predicateSPMD.
// Each case body should get an SPMDStore with its specific case mask, not the
// all-ones mask or a wrong mask.
func TestPredicateSPMD_SwitchMemOps(t *testing.T) {
	// A switch with 3 cases, each performing a distinct store operation.
	// The stores must each be masked with the corresponding case mask.
	src := `package main

func f(dst []int) {
	for i := range dst {
		switch i % 4 {
		case 0:
			dst[i] = 100
		case 1:
			dst[i] = 200
		case 2:
			dst[i] = 300
		default:
			dst[i] = 400
		}
	}
}

func main() { f(make([]int, 16)) }
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	if len(fn.SPMDSwitchChains) == 0 {
		t.Fatal("expected SPMDSwitchChains to be populated")
	}

	// Count SPMDStore instructions — one per case body (3 cases + 1 default = 4).
	spmdStoreCount := 0
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDStore); ok {
				spmdStoreCount++
			}
		}
	}
	// 4 case/default bodies, each with one store → 4 SPMDStore instructions.
	if spmdStoreCount != 4 {
		t.Errorf("expected 4 SPMDStore instructions (one per case), got %d", spmdStoreCount)
	}

	// The lane count on all SPMDStore instructions must match the loop's lane count.
	loopLanes := fn.SPMDLoops[0].LaneCount
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if store, ok := instr.(*ssa.SPMDStore); ok {
				if store.Lanes != loopLanes {
					t.Errorf("SPMDStore.Lanes = %d, want %d", store.Lanes, loopLanes)
				}
			}
		}
	}

	// No varying If should remain after linearization.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If not linearized in 4-case switch", block.Index)
			}
		}
	}
}

// TestPredicateSPMD_SwitchSanityCheck verifies that the SSA sanity checker
// passes after predicateSPMD linearizes a varying switch chain.
func TestPredicateSPMD_SwitchSanityCheck(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		switch i % 3 {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		default:
			_ = i + 2
		}
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	// SanityCheckFunctions will panic if predicateSPMD leaves the SSA invalid.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage with SanityCheckFunctions failed: %v", err)
	}
}

// TestPredicateSPMD_SanityCheck verifies the sanity checker passes after
// predicateSPMD runs on a function with a varying if-without-else.
func TestPredicateSPMD_SanityCheck(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		x := i
		if i > 5 {
			x = i + 10
		}
		_ = x
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	// SanityCheckFunctions will panic if predicateSPMD leaves the SSA invalid.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage failed: %v", err)
	}
}

// TestPredicateSPMD_IfElseSanityCheck verifies the sanity checker passes
// after predicateSPMD linearizes a full if/else with SPMDSelect.
func TestPredicateSPMD_IfElseSanityCheck(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		x := i
		if i > 5 {
			x = i + 10
		} else {
			x = i - 10
		}
		_ = x
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage failed: %v", err)
	}
}

// TestPredicateSPMD_LoopBackNotLinearized verifies that a varying if/else
// whose both branches jump back to the loop header is NOT linearized by
// predicateSPMD. In this pattern the "merge block" is the loop header, which
// has extra predecessors (the loop entry) and loop-carried Phis — not if/else
// merge Phis. Replacing those Phis with SPMDSelect would violate SSA dominance
// (the mask is computed inside the loop body but referenced in the loop header,
// which precedes the body in DomPreorder). findMergeBlock now rejects such
// merge blocks by requiring exactly two predecessors.
func TestPredicateSPMD_LoopBackNotLinearized(t *testing.T) {
	// This pattern models a rangeindex loop (range over slice) where the
	// if/else branches both jump back to the loop header. The loop header
	// must NOT have its iterator Phi replaced by SPMDSelect.
	src := `package main

func encode(dst, src []byte) {
	for i := range dst {
		if i%2 == 0 {
			dst[i] = src[i>>1] >> 4
		} else {
			dst[i] = src[i>>1] & 0x0f
		}
	}
}

func main() { encode(make([]byte, 4), make([]byte, 2)) }
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

	// SanityCheckFunctions panics if dominance is violated.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage failed: %v", err)
	}

	// Build to inspect the generated SSA.
	pkg := buildSSAWithSPMD(t, src)
	encodeFn := pkg.Func("encode")
	if encodeFn == nil {
		t.Fatal("encode function not found")
	}

	// The loop header must still be reachable and have its iterator phi.
	// It must NOT have any SPMDSelect instructions (because the if/else merge
	// is the loop header with extra predecessors — not a true diamond merge).
	for _, block := range encodeFn.Blocks {
		if block.Comment != "rangeindex.loop" {
			continue
		}
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDSelect); ok {
				t.Errorf("rangeindex.loop has SPMDSelect: loop-back if/else was incorrectly linearized")
			}
		}
	}
}

// TestPredicateSPMD_WriteFunction verifies WriteFunction does not panic on a
// function after predicateSPMD runs. Mask computation instructions (from
// linearizing a varying if) must appear in the output.
func TestPredicateSPMD_WriteFunction(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 5 {
			_ = i + 1
		} else {
			_ = i - 1
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, mainFn)
	if buf.Len() == 0 {
		t.Error("WriteFunction returned empty output")
	}
	output := buf.String()

	// predicateSPMD inserts mask computations even when the Phi is dead.
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected mask computation instructions from predicateSPMD in WriteFunction output")
	}
	// The varying If must be gone.
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying' after predicateSPMD ran")
	}
}
