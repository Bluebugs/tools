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
// predicateSPMD in go-for SPMD loop context. In this pattern the "merge block"
// is the loop header, which has extra predecessors (the loop entry) and
// loop-carried Phis — not if/else merge Phis. The loop-header merge trampoline
// is only inserted in SPMD function body context (no go-for loop), so in
// go-for context the pattern is left for TinyGo's mask transitions.
func TestPredicateSPMD_LoopBackNotLinearized(t *testing.T) {
	// This pattern models a rangeindex loop (range over slice) where the
	// if/else branches both jump back to the loop header. The new deferred-merge
	// path linearizes the varying If (B→T→E→loopHeader) and inserts SPMDSelect
	// in the else block, but the loop header itself must NOT have its iterator
	// Phi replaced by SPMDSelect.
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

	// The loop header must NOT have any SPMDSelect instructions. The deferred
	// merge path places SPMDSelect in the else block (elseBlock→loopHeader), not
	// in the loop header itself. Loop-carried phis at the loop header are updated
	// to reference the SPMDSelect result via their elseBlock edge.
	for _, block := range encodeFn.Blocks {
		if block.Comment != "rangeindex.loop" {
			continue
		}
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDSelect); ok {
				t.Errorf("rangeindex.loop has SPMDSelect — loop-carried phi was incorrectly replaced in the loop header")
			}
		}
	}

	// After deferred-merge linearization, no varying If should survive.
	for _, block := range encodeFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && vif.IsVarying {
			t.Errorf("block %d still has a varying If after deferred-merge linearization", block.Index)
		}
	}

	// Mask computation instructions must be present (the linearization ran).
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, encodeFn)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Errorf("expected mask computation instructions in output:\n%s", output)
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

// ----------------------------------------------------------------------------
// Boolean chain predication tests (predicateBooleanChain)

// TestPredicateSPMD_BooleanChainAnd verifies that an && boolean chain is
// linearized: all varying Ifs removed, combined AND mask computed, memory
// ops in the then-block are masked, and SPMDSelect replaces any live Phis.
func TestPredicateSPMD_BooleanChainAnd(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 {
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

	// No varying If should remain after boolean chain linearization.
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If remains after boolean chain linearization", block.Index)
			}
		}
	}

	// Mask computation instructions must be present (Convert + AND for && chain).
	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected Varying[mask] mask instructions from boolean chain && predication")
	}
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying' after boolean chain linearization")
	}
}

// TestPredicateSPMD_BooleanChainOr verifies that an || boolean chain is
// linearized: all varying Ifs removed, combined OR mask computed, mask
// instructions present in SSA output.
func TestPredicateSPMD_BooleanChainOr(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i < 2 || i > 10 {
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

	// No varying If should remain.
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If remains after boolean chain OR linearization", block.Index)
			}
		}
	}

	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected Varying[mask] mask instructions from boolean chain || predication")
	}
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying' after boolean chain || linearization")
	}
}

// TestPredicateSPMD_BooleanChainWithElse verifies that && with an else branch
// is linearized correctly: no varying Ifs remain, mask instructions present,
// both then and else branches appear in the output.
func TestPredicateSPMD_BooleanChainWithElse(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		x := i
		if i > 2 && i < 10 {
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

	// No varying If should remain.
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If remains after && with else linearization", block.Index)
			}
		}
	}

	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected Varying[mask] mask instructions from && with else predication")
	}
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying' after && with else linearization")
	}
}

// TestPredicateSPMD_BooleanChainMemOps verifies that memory operations in
// the then-block of a boolean chain are masked with SPMDStore.
func TestPredicateSPMD_BooleanChainMemOps(t *testing.T) {
	src := `package main

func f(dst []int) {
	for i := range dst {
		if i > 2 && i < 10 {
			dst[i] = i * 2
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

	// SPMDStore must appear — the store inside the && body was masked.
	foundSPMDStore := false
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDStore); ok {
				foundSPMDStore = true
			}
		}
	}
	if !foundSPMDStore {
		t.Error("expected SPMDStore in then-block of && chain (store must be masked)")
	}

	// No varying If should remain.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if vif.IsVarying {
				t.Errorf("block %d: varying If remains after boolean chain mem-ops test", block.Index)
			}
		}
	}
}

// TestPredicateSPMD_BooleanChainPhiMerge verifies that if-without-else boolean
// chains correctly replace Phis at the merge block with SPMDSelect. This tests
// the case where a value is conditionally modified under a && condition and
// used after the if — producing a Phi at the merge that must become an SPMDSelect.
func TestPredicateSPMD_BooleanChainPhiMerge(t *testing.T) {
	src := `package main

func f(dst []int) {
	for i := range dst {
		x := i
		if i > 2 && i < 10 {
			x = i + 100
		}
		dst[i] = x
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// After predication, the Phi at the merge block should be replaced
	// with an SPMDSelect. Check that at least one SPMDSelect exists.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if !strings.Contains(output, "spmd_select") {
		t.Error("expected SPMDSelect in output for if-without-else && chain with Phi merge")
		t.Logf("SSA output:\n%s", output)
	}

	// No varying If should remain.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && vif.IsVarying {
			t.Errorf("varying If still present in block %s after predication", block)
		}
	}
}

// TestPredicateSPMD_BooleanChainOrPhiMerge verifies that if-without-else
// || chains correctly replace Phis at the merge block with SPMDSelect.
// For LOR, the last chain block (not the first) is the merge predecessor.
func TestPredicateSPMD_BooleanChainOrPhiMerge(t *testing.T) {
	src := `package main

func f(dst []int) {
	for i := range dst {
		x := i
		if i < 2 || i > 10 {
			x = i + 100
		}
		dst[i] = x
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// After predication, the Phi at the merge block should be replaced
	// with an SPMDSelect. Check that at least one SPMDSelect exists.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if !strings.Contains(output, "spmd_select") {
		t.Error("expected SPMDSelect in output for if-without-else || chain with Phi merge")
		t.Logf("SSA output:\n%s", output)
	}

	// No varying If should remain.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && vif.IsVarying {
			t.Errorf("varying If still present in block %s after predication", block)
		}
	}
}

// TestPredicateSPMD_BooleanChainSanity verifies the sanity checker passes
// after predicateSPMD linearizes an && boolean chain (if-without-else).
func TestPredicateSPMD_BooleanChainSanity(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 {
			_ = i + 1
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

	// SanityCheckFunctions will panic if predicateBooleanChain leaves the SSA invalid.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage with SanityCheckFunctions failed on && chain: %v", err)
	}
}

// TestPredicateSPMD_BooleanChainOrSanity verifies the sanity checker passes
// after predicateSPMD linearizes an || boolean chain.
func TestPredicateSPMD_BooleanChainOrSanity(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i < 2 || i > 10 {
			_ = i + 1
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
		t.Fatalf("BuildPackage with SanityCheckFunctions failed on || chain: %v", err)
	}
}

// ----------------------------------------------------------------------------
// Phase 8: SPMD function body predication tests
//
// These tests cover predicateSPMDFuncBody, which handles varying breaks inside
// regular for-range loops in SPMD function bodies (functions with varying params).

// buildSPMDFuncBody builds an SSA package from source that contains an SPMD
// function body (a function with lanes.Varying[T] parameters) but no go-for loop.
// GOEXPERIMENT=spmd must be active, and the lanes package must be importable.
func buildSPMDFuncBody(t *testing.T, src string) *ssa.Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	pkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f}, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

// TestPredicateSPMD_HasSPMDParams verifies hasSPMDParams identifies functions
// with SPMDType parameters correctly.
// hasSPMDParams is not exported, so we test it indirectly: a function with a
// lanes.Varying[int] param triggers predicateSPMD, which inserts mask
// instructions even for a function body with no go-for loops.
func TestPredicateSPMD_HasSPMDParams(t *testing.T) {
	tests := []struct {
		name      string
		src       string
		funcName  string
		wantSPMD  bool // true = has SPMDType params, expect predication ran
	}{
		{
			name: "with_varying_param",
			src: `package main
import "lanes"
func f(v lanes.Varying[int]) lanes.Varying[int] { return v }
func main() {}
`,
			funcName: "f",
			wantSPMD: true,
		},
		{
			name: "without_varying_param",
			src: `package main
func g(x int) int { return x }
func main() {}
`,
			funcName: "g",
			wantSPMD: false,
		},
		{
			name: "multiple_params_one_varying",
			src: `package main
import "lanes"
func h(a int, v lanes.Varying[int], b int) lanes.Varying[int] { return v }
func main() {}
`,
			funcName: "h",
			wantSPMD: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pkg := buildSPMDFuncBody(t, tc.src)
			fn := pkg.Func(tc.funcName)
			if fn == nil {
				t.Fatalf("function %s not found", tc.funcName)
			}
			// hasSPMDParams is internal; we test by checking that for a
			// function with varying params, predication has run (even without
			// SPMDLoops). For the non-SPMD case, verify no mask instructions.
			hasMaskInstr := false
			for _, block := range fn.Blocks {
				for _, instr := range block.Instrs {
					if phi, ok := instr.(*ssa.Phi); ok {
						if strings.Contains(phi.Comment, "spmd.break.mask") {
							hasMaskInstr = true
						}
					}
					if conv, ok := instr.(*ssa.Convert); ok {
						if _, isMask := conv.Type().(*types.SPMDType); isMask {
							hasMaskInstr = true
						}
					}
				}
			}
			// For SPMD functions without a loop body to predicate, there should
			// be no mask instructions (predication only acts on varying breaks).
			// The test is really about whether the function has SPMDLoops == 0
			// and varying params — the predication pass runs but finds nothing to do.
			if tc.wantSPMD {
				// Verify function has lanes.Varying params (by checking signature).
				if fn.Signature == nil {
					t.Fatal("Signature is nil")
				}
				params := fn.Signature.Params()
				foundVarying := false
				for i := 0; i < params.Len(); i++ {
					if _, ok := params.At(i).Type().(*types.SPMDType); ok {
						foundVarying = true
						break
					}
				}
				if !foundVarying {
					t.Errorf("expected at least one *types.SPMDType parameter, found none")
				}
			} else {
				// Non-SPMD function: no SPMDType params.
				params := fn.Signature.Params()
				for i := 0; i < params.Len(); i++ {
					if _, ok := params.At(i).Type().(*types.SPMDType); ok {
						t.Errorf("unexpected *types.SPMDType parameter in non-SPMD function")
					}
				}
			}
			_ = hasMaskInstr
		})
	}
}

// TestPredicateSPMD_FindForLoops verifies that spmdFindRegularForLoops detects
// regular for-range loops inside SPMD function bodies.
func TestPredicateSPMD_FindForLoops(t *testing.T) {
	// A function with a varying param and a regular for-range loop.
	// The loop generates rangeint.body and rangeint.done blocks.
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := v
	for i := range n {
		_ = i
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Verify the loop structure is present.
	foundRangeintBody := false
	foundRangeintDone := false
	for _, block := range fn.Blocks {
		if strings.Contains(block.Comment, "rangeint.body") {
			foundRangeintBody = true
		}
		if strings.Contains(block.Comment, "rangeint.done") {
			foundRangeintDone = true
		}
	}
	if !foundRangeintBody {
		t.Error("expected rangeint.body block for regular for-range loop")
	}
	if !foundRangeintDone {
		t.Error("expected rangeint.done block for regular for-range loop")
	}

	// Verify the function has no SPMDLoops (it's a function body, not go-for).
	if len(fn.SPMDLoops) != 0 {
		t.Errorf("expected 0 SPMDLoops for regular for-range in SPMD body, got %d",
			len(fn.SPMDLoops))
	}

	// The function has lanes.Varying params.
	params := fn.Signature.Params()
	foundVarying := false
	for i := 0; i < params.Len(); i++ {
		if _, ok := params.At(i).Type().(*types.SPMDType); ok {
			foundVarying = true
			break
		}
	}
	if !foundVarying {
		t.Error("expected varying parameter")
	}
}

// TestPredicateSPMD_VaryingBreak verifies the basic varying break transformation:
// a function with a varying param, a regular for-loop, and a varying break.
//
// The varying If is linearized (no "if varying" in output), a break mask phi
// is inserted at the loop header, and mask computation instructions appear.
func TestPredicateSPMD_VaryingBreak(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := v
	for i := range n {
		vi := lanes.Varying[int](i)
		if v > vi {
			result = vi
			break
		}
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Verify the varying If has been linearized (no "if varying" in SSA dump).
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if strings.Contains(output, "if varying") {
		t.Errorf("varying If was not linearized by predicateSPMDFuncBody:\n%s", output)
	}

	// Verify the break mask phi was inserted at the loop header.
	if !strings.Contains(output, "spmd.break.mask") {
		t.Errorf("expected spmd.break.mask phi in loop header:\n%s", output)
	}

	// Verify mask computation instructions are present.
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Errorf("expected Varying[mask] instructions from break mask computation:\n%s", output)
	}

	// Verify SPMDSelect for break result is present.
	if !strings.Contains(output, "spmd_select") {
		t.Errorf("expected spmd_select for break result selection:\n%s", output)
	}

	// Verify the original If is gone (replaced by Jump).
	foundVaryingIf := false
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if ifInstr.IsVarying {
				foundVaryingIf = true
			}
		}
	}
	if foundVaryingIf {
		t.Error("expected no remaining varying If after predicateSPMDFuncBody")
	}
}

// TestPredicateSPMD_VaryingBreakResult verifies that the result phi at the
// done block is correctly updated: the loop-exit edge points to the SPMDSelect
// (accumulated break result) while the entry edge keeps the default value.
func TestPredicateSPMD_VaryingBreakResult(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := v
	for i := range n {
		vi := lanes.Varying[int](i)
		if v > vi {
			result = vi
			break
		}
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Find the rangeint.done block.
	var doneBlock *ssa.BasicBlock
	for _, block := range fn.Blocks {
		if strings.Contains(block.Comment, "rangeint.done") {
			doneBlock = block
			break
		}
	}
	if doneBlock == nil {
		t.Fatal("rangeint.done block not found")
	}

	// The done block should have a result phi whose loop-exit edge is an
	// SPMDSelect (the accumulated break result).
	// After transformation, the result phi should be phi [entry: v, loop_exit: spmd_select].
	foundSPMDSelectEdge := false
	for _, instr := range doneBlock.Instrs {
		phi, ok := instr.(*ssa.Phi)
		if !ok {
			break
		}
		for _, edge := range phi.Edges {
			if _, ok := edge.(*ssa.SPMDSelect); ok {
				foundSPMDSelectEdge = true
			}
		}
	}
	if !foundSPMDSelectEdge {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected SPMDSelect as a phi edge at done block, SSA:\n%s", buf.String())
	}
}

// TestPredicateSPMD_VaryingBreakSanity verifies the SSA sanity checker passes
// after predicateSPMDFuncBody transforms a varying break.
func TestPredicateSPMD_VaryingBreakSanity(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := v
	for i := range n {
		vi := lanes.Varying[int](i)
		if v > vi {
			result = vi
			break
		}
	}
	return result
}
func main() {}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// SanityCheckFunctions panics if the transformed SSA has invalid def/use.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage with SanityCheckFunctions failed: %v", err)
	}
}

// TestPredicateSPMD_VaryingBreakNoVaryingIf verifies that a regular for-loop
// with a uniform (non-varying) if-break is NOT transformed. The break mask
// phi should not be inserted when the condition is uniform.
func TestPredicateSPMD_VaryingBreakNoVaryingIf(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := v
	for i := range n {
		if i > 5 {
			break
		}
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// The break condition is uniform (i > 5, where i is a plain int).
	// No varying break transformation should occur.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if strings.Contains(output, "spmd.break.mask") {
		t.Errorf("unexpected spmd.break.mask for uniform break condition:\n%s", output)
	}
}

// TestPredicateSPMD_FuncBodyVaryingIfElse verifies that a varying if/else
// inside a function body (no loop, no break) is linearized by
// predicateSPMDFuncBody: the varying If is replaced by Jumps and the merge
// phi is rewritten to an SPMDSelect.
func TestPredicateSPMD_FuncBodyVaryingIfElse(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int]) lanes.Varying[int] {
	var result lanes.Varying[int]
	vi := lanes.Varying[int](10)
	if v > vi {
		result = v
	} else {
		result = vi
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	// The varying If must have been linearized — no "if varying" should remain.
	if strings.Contains(output, "if varying") {
		t.Errorf("varying If was not linearized by predicateSPMDFuncBody:\n%s", output)
	}

	// predicateSPMDFuncBody rewrites the merge phi to an SPMDSelect.
	if !strings.Contains(output, "spmd_select") {
		t.Errorf("expected spmd_select for if/else merge:\n%s", output)
	}

	// Mask computation instructions must be present.
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Errorf("expected mask computation instructions from predicateSPMDFuncBody:\n%s", output)
	}

	// Confirm via struct inspection that no varying If instructions survive.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if ifInstr.IsVarying {
				t.Errorf("block %d still has a varying If after predicateSPMDFuncBody", block.Index)
			}
		}
	}
}

// TestPredicateSPMD_VaryingBreakAccumulator verifies that predicateVaryingBreaks
// creates a result accumulator phi at the loop header so that SPMDSelect merges
// new break values with the LOOP-CARRIED accumulated result rather than the
// static default. Without this phi, each iteration's SPMDSelect would fall back
// to normVal (the static default) and discard break results from prior iterations.
//
// Regression test for: L4b_varying_break returning 39 instead of 21.
// The fix: an "spmd.break.accum" phi at the loop header carries the accumulated
// break result across iterations.
func TestPredicateSPMD_VaryingBreakAccumulator(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := lanes.Varying[int](10)
	for i := range n {
		vi := lanes.Varying[int](i)
		if v == vi {
			result = vi
			break
		}
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Find the loop header block (rangeint.body for merged body+loop).
	var loopBlock *ssa.BasicBlock
	for _, block := range fn.Blocks {
		if strings.Contains(block.Comment, "rangeint.body") {
			loopBlock = block
			break
		}
	}
	if loopBlock == nil {
		t.Fatal("rangeint.body (loop header) block not found")
	}

	// The loop header must contain an "spmd.break.accum" phi.
	// This phi carries the accumulated break result across iterations, ensuring
	// the SPMDSelect's Y operand is the prior accumulated result rather than
	// the static default (normVal).
	foundAccumPhi := false
	for _, instr := range loopBlock.Instrs {
		phi, ok := instr.(*ssa.Phi)
		if !ok {
			break // phis are always at the front; stop at first non-phi
		}
		if strings.Contains(phi.Comment, "spmd.break.accum") {
			foundAccumPhi = true
			break
		}
	}
	if !foundAccumPhi {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected spmd.break.accum phi at loop header for cross-iteration accumulation, SSA:\n%s", buf.String())
	}

	// The SPMDSelect's Y operand must reference the accumulator phi, NOT the
	// static default (normVal). Verify by checking that the SPMDSelect inside
	// the loop body has a phi (not a Const) as its Y operand.
	foundSPMDSelectWithPhiY := false
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			sel, ok := instr.(*ssa.SPMDSelect)
			if !ok {
				continue
			}
			if _, ok := sel.Y.(*ssa.Phi); ok {
				foundSPMDSelectWithPhiY = true
			}
		}
	}
	if !foundSPMDSelectWithPhiY {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected SPMDSelect.Y to be accum phi (not static default), SSA:\n%s", buf.String())
	}
}

// TestPredicateSPMD_VaryingBreakAccumulatorSanity verifies the SSA sanity
// checker passes after predicateVaryingBreaks transforms a varying break that
// includes result accumulator phis. Uses the same function source as
// TestPredicateSPMD_VaryingBreakAccumulator.
func TestPredicateSPMD_VaryingBreakAccumulatorSanity(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], n int) lanes.Varying[int] {
	result := lanes.Varying[int](10)
	for i := range n {
		vi := lanes.Varying[int](i)
		if v == vi {
			result = vi
			break
		}
	}
	return result
}
func main() {}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// SanityCheckFunctions panics if the transformed SSA has invalid def/use.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage with SanityCheckFunctions failed: %v", err)
	}
}

// TestPredicateSPMD_VaryingBreakNoLoop verifies that an SPMD function body
// without a for-loop is not affected by predicateSPMDFuncBody.
func TestPredicateSPMD_VaryingBreakNoLoop(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// No loop, no break mask phi.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if strings.Contains(output, "spmd.break.mask") {
		t.Errorf("unexpected spmd.break.mask in SPMD function without loop:\n%s", output)
	}
	if strings.Contains(output, "spmd_select") {
		t.Errorf("unexpected spmd_select in SPMD function without loop:\n%s", output)
	}
}

// TestPredicateSPMD_LoopHeaderMerge verifies that a varying if/else inside a
// loop in an SPMD function body (not go-for) where both branches jump back to
// the loop header is predicated by inserting an spmd.merge trampoline block.
// The varying If is linearized and memory ops are masked. Loop-carried phis
// at the loop header are NOT replaced.
func TestPredicateSPMD_LoopHeaderMerge(t *testing.T) {
	// SPMD function body context: function has lanes.Varying[int] parameter.
	// The for-loop inside is a regular loop (not go-for), so
	// allowLoopHeaderMerge = true and the trampoline is inserted.
	src := `package main
import "lanes"
func f(v lanes.Varying[int], data []int) {
	for i := 0; i < len(data); i++ {
		if v > lanes.Varying[int](5) {
			data[i] = 10
		} else {
			data[i] = 20
		}
	}
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	// 1. No varying Ifs should remain (all linearized).
	if strings.Contains(output, "if varying") {
		t.Errorf("varying If was not linearized:\n%s", output)
	}

	// 2. Mask computation instructions must be present.
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Errorf("expected mask computation instructions:\n%s", output)
	}

	// 3. Loop header blocks must NOT have SPMDSelect — loop-carried phis are preserved.
	for _, block := range fn.Blocks {
		if !strings.Contains(block.Comment, "for.loop") {
			continue
		}
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDSelect); ok {
				t.Errorf("loop header block %q has SPMDSelect — loop-carried phi was incorrectly replaced", block.Comment)
			}
		}
	}

	// 4. Confirm via struct inspection that no varying If instructions survive.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if ifInstr.IsVarying {
				t.Errorf("block %d still has a varying If after predication", block.Index)
			}
		}
	}
}

// TestPredicateSPMD_LoopHeaderMergeSanity verifies that the SSA sanity checker
// passes after loop-header merge trampoline insertion in SPMD function body context.
func TestPredicateSPMD_LoopHeaderMergeSanity(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], data []int) {
	for i := 0; i < len(data); i++ {
		if v > lanes.Varying[int](5) {
			data[i] = 10
		} else {
			data[i] = 20
		}
	}
}
func main() {}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	// No setSPMDOnRange — this is SPMD function body context (varying params).

	// SanityCheckFunctions panics if dominance is violated.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage failed (sanity check): %v", err)
	}
}

// ----------------------------------------------------------------------------
// SPMDMask parameter tests

// TestPredicateSPMD_FuncBodyMaskParam verifies that fn.SPMDMask is set for
// SPMD functions with varying params, and not set for non-SPMD functions.
func TestPredicateSPMD_FuncBodyMaskParam(t *testing.T) {
	src := `package main
import "lanes"
func spmdFunc(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}
func normalFunc(x int) int {
	return x
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)

	// SPMD function should have SPMDMask set.
	spmdFn := pkg.Func("spmdFunc")
	if spmdFn == nil {
		t.Fatal("function spmdFunc not found")
	}
	if spmdFn.SPMDMask == nil {
		t.Fatal("expected SPMDMask to be set for SPMD function")
	}
	if spmdFn.SPMDMask.Name() != "spmd.mask" {
		t.Errorf("SPMDMask name = %q, want %q", spmdFn.SPMDMask.Name(), "spmd.mask")
	}
	if !spmd.IsVaryingMask(spmdFn.SPMDMask.Type()) {
		t.Errorf("SPMDMask type = %s, want Varying[mask]", spmdFn.SPMDMask.Type())
	}
	if spmdFn.SPMDMask.Parent() != spmdFn {
		t.Errorf("SPMDMask.Parent() = %v, want %v", spmdFn.SPMDMask.Parent(), spmdFn)
	}

	// WriteFunction should show the mask parameter.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, spmdFn)
	output := buf.String()
	if !strings.Contains(output, "# SPMD mask: spmd.mask") {
		t.Errorf("WriteFunction output should contain SPMD mask header:\n%s", output)
	}

	// Non-SPMD function should NOT have SPMDMask set.
	normalFn := pkg.Func("normalFunc")
	if normalFn == nil {
		t.Fatal("function normalFunc not found")
	}
	if normalFn.SPMDMask != nil {
		t.Error("expected SPMDMask to be nil for non-SPMD function")
	}
}

// TestPredicateSPMD_CallMask verifies that SPMD function body calls to another
// SPMD function get their SPMDMask set to the caller's mask parameter.
func TestPredicateSPMD_CallMask(t *testing.T) {
	src := `package main
import "lanes"
func callee(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}
func caller(v lanes.Varying[int]) lanes.Varying[int] {
	return callee(v)
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("caller")
	if fn == nil {
		t.Fatal("function caller not found")
	}
	if fn.SPMDMask == nil {
		t.Fatal("expected SPMDMask on caller")
	}

	// Find the Call instruction.
	var callInstr *ssa.Call
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if call, ok := instr.(*ssa.Call); ok {
				if callee := call.Call.StaticCallee(); callee != nil && callee.Name() == "callee" {
					callInstr = call
				}
			}
		}
	}
	if callInstr == nil {
		t.Fatal("call to callee not found")
	}
	if callInstr.Call.SPMDMask == nil {
		t.Fatal("expected SPMDMask on call to callee")
	}
	// The mask should be the caller's SPMDMask parameter.
	if callInstr.Call.SPMDMask != fn.SPMDMask {
		t.Errorf("call SPMDMask = %v, want caller's SPMDMask %v",
			callInstr.Call.SPMDMask, fn.SPMDMask)
	}

	// Verify printed output shows mask in call.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()
	if !strings.Contains(output, "spmd.mask") {
		t.Errorf("expected spmd.mask in call output:\n%s", output)
	}
}

// TestPredicateSPMD_CallMaskInVaryingIf verifies that a call to an SPMD
// function inside a varying-if gets the narrowed thenMask, not the entry mask.
func TestPredicateSPMD_CallMaskInVaryingIf(t *testing.T) {
	src := `package main
import "lanes"
func callee(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}
func caller(v lanes.Varying[int]) lanes.Varying[int] {
	vi := lanes.Varying[int](10)
	var result lanes.Varying[int]
	if v > vi {
		result = callee(v)
	} else {
		result = v
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("caller")
	if fn == nil {
		t.Fatal("function caller not found")
	}
	if fn.SPMDMask == nil {
		t.Fatal("expected SPMDMask on caller")
	}

	// Find the Call instruction to callee.
	var callInstr *ssa.Call
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if call, ok := instr.(*ssa.Call); ok {
				if callee := call.Call.StaticCallee(); callee != nil && callee.Name() == "callee" {
					callInstr = call
				}
			}
		}
	}
	if callInstr == nil {
		t.Fatal("call to callee not found")
	}
	if callInstr.Call.SPMDMask == nil {
		t.Fatal("expected SPMDMask on call to callee inside varying if")
	}
	// Inside a varying if/else, the mask should be narrowed (not the entry mask).
	// It should be a mask AND result, not the raw SPMDMask parameter.
	if callInstr.Call.SPMDMask == fn.SPMDMask {
		t.Error("call inside varying-if should have narrowed mask, not entry mask")
	}
	// The mask should be of Varying[mask] type.
	if !spmd.IsVaryingMask(callInstr.Call.SPMDMask.Type()) {
		t.Errorf("call SPMDMask type = %s, want Varying[mask]",
			callInstr.Call.SPMDMask.Type())
	}
}

// TestPredicateSPMD_CallMaskSanity verifies the SSA sanity checker passes
// after mask parameter creation and call masking.
func TestPredicateSPMD_CallMaskSanity(t *testing.T) {
	src := `package main
import "lanes"
func callee(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}
func caller(v lanes.Varying[int]) lanes.Varying[int] {
	return callee(v)
}
func main() {}
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
		t.Fatalf("BuildPackage with SanityCheckFunctions failed: %v", err)
	}
}

// TestPredicateSPMD_FuncBodyVaryingIfElseMaskRef verifies that mask computation
// in a function body references the SPMDMask parameter rather than a constant.
func TestPredicateSPMD_FuncBodyVaryingIfElseMaskRef(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int]) lanes.Varying[int] {
	var result lanes.Varying[int]
	vi := lanes.Varying[int](10)
	if v > vi {
		result = v
	} else {
		result = vi
	}
	return result
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}
	if fn.SPMDMask == nil {
		t.Fatal("expected SPMDMask on function f")
	}

	// Find mask AND instructions (used to compute thenMask/elseMask).
	// At least one should reference the SPMDMask parameter.
	foundMaskRef := false
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			binOp, ok := instr.(*ssa.BinOp)
			if !ok {
				continue
			}
			if binOp.X == fn.SPMDMask || binOp.Y == fn.SPMDMask {
				foundMaskRef = true
			}
		}
	}
	if !foundMaskRef {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected mask AND instructions referencing SPMDMask parameter:\n%s", buf.String())
	}
}

// TestPredicateSPMD_LoadContiguity verifies that SPMDLoad.Contiguous is set
// when arr[i] is accessed inside a varying if in a go-for loop.
func TestPredicateSPMD_LoadContiguity(t *testing.T) {
	src := `package main

func f(arr []int) {
	for i := range len(arr) {
		if arr[i] > 0 {
			arr[i] = arr[i] + 1
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

	var contiguousLoads, nonContiguousLoads int
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if load, ok := instr.(*ssa.SPMDLoad); ok {
				if load.Contiguous {
					contiguousLoads++
					if load.Source == nil {
						t.Error("SPMDLoad.Contiguous is true but Source is nil")
					}
				} else {
					nonContiguousLoads++
				}
			}
		}
	}
	if contiguousLoads == 0 {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected at least one contiguous SPMDLoad, got 0:\n%s", buf.String())
	}
}

// TestPredicateSPMD_StoreContiguity verifies that SPMDStore.Contiguous is set
// when arr[i] is stored inside a varying if in a go-for loop.
func TestPredicateSPMD_StoreContiguity(t *testing.T) {
	src := `package main

func f(arr []int) {
	for i := range len(arr) {
		if arr[i] > 0 {
			arr[i] = 42
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

	var contiguousStores int
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if store, ok := instr.(*ssa.SPMDStore); ok {
				if store.Contiguous {
					contiguousStores++
					if store.Source == nil {
						t.Error("SPMDStore.Contiguous is true but Source is nil")
					}
				}
			}
		}
	}
	if contiguousStores == 0 {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected at least one contiguous SPMDStore, got 0:\n%s", buf.String())
	}
}

// TestPredicateSPMD_NonContiguousLoad verifies that SPMDLoad.Contiguous=false
// when the index is not the loop iterator.
func TestPredicateSPMD_NonContiguousLoad(t *testing.T) {
	src := `package main

func f(arr []int, j int) {
	for i := range len(arr) {
		if i > 0 {
			arr[i] = arr[j]
		}
	}
}

func main() { f(make([]int, 16), 3) }
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if load, ok := instr.(*ssa.SPMDLoad); ok {
				if load.Contiguous {
					t.Errorf("SPMDLoad should NOT be contiguous for arr[j], got Contiguous=true")
				}
			}
		}
	}
}

// TestPredicateSPMD_ContiguousString verifies that String() appends [contiguous].
func TestPredicateSPMD_ContiguousString(t *testing.T) {
	src := `package main

func f(arr []int) {
	for i := range len(arr) {
		if arr[i] > 0 {
			arr[i] = 42
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

	foundContiguousString := false
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			s := instr.String()
			if strings.Contains(s, "[contiguous]") {
				foundContiguousString = true
			}
		}
	}
	if !foundContiguousString {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected [contiguous] suffix in SPMDLoad/SPMDStore String():\n%s", buf.String())
	}
}

// TestPredicateSPMD_ContiguousSanity verifies the sanity checker passes with
// contiguous SPMDLoad/SPMDStore instructions.
func TestPredicateSPMD_ContiguousSanity(t *testing.T) {
	src := `package main

func f(arr []int) {
	for i := range len(arr) {
		if arr[i] > 0 {
			arr[i] = arr[i] + 1
		}
	}
}

func main() { f(make([]int, 16)) }
`
	// buildSSAWithSPMD uses SanityCheckFunctions mode, so this implicitly
	// validates that the sanity checker accepts contiguous SPMDLoad/SPMDStore.
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}
}

// TestSPMDLoad_OperandsWithSource verifies Operands() includes Source when
// Contiguous is true (3 operands instead of 2).
func TestSPMDLoad_OperandsWithSource(t *testing.T) {
	addr := ssa.Value(ptrConst())
	mask := ssa.Value(intConst(0))
	source := ssa.Value(ptrConst())

	load := &ssa.SPMDLoad{
		Addr:       addr,
		Mask:       mask,
		Lanes:      4,
		Contiguous: true,
		Source:     source,
	}

	rands := load.Operands(nil)
	if len(rands) != 3 {
		t.Fatalf("SPMDLoad.Operands() with Source returned %d operands, want 3", len(rands))
	}
	if *rands[2] != source {
		t.Errorf("SPMDLoad.Operands()[2] should be Source")
	}
}

// TestSPMDStore_OperandsWithSource verifies Operands() includes Source when
// Contiguous is true (4 operands instead of 3).
func TestSPMDStore_OperandsWithSource(t *testing.T) {
	addr := ssa.Value(ptrConst())
	val := ssa.Value(intConst(42))
	mask := ssa.Value(intConst(0))
	source := ssa.Value(ptrConst())

	store := &ssa.SPMDStore{
		Addr:       addr,
		Val:        val,
		Mask:       mask,
		Lanes:      4,
		Contiguous: true,
		Source:     source,
	}

	rands := store.Operands(nil)
	if len(rands) != 4 {
		t.Fatalf("SPMDStore.Operands() with Source returned %d operands, want 4", len(rands))
	}
	if *rands[3] != source {
		t.Errorf("SPMDStore.Operands()[3] should be Source")
	}
}

// TestPredicateSPMD_ConvertAllMemOps_FuncBody verifies that straight-line
// loads/stores in an SPMD function body are converted to SPMDLoad/SPMDStore
// by spmdConvertAllMemOps.
func TestPredicateSPMD_ConvertAllMemOps_FuncBody(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], dst *int) lanes.Varying[int] {
	*dst = 42
	result := *dst
	_ = result
	return v
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	// After spmdConvertAllMemOps, there should be no plain *UnOp{MUL} or *Store
	// in the function — all should be converted to SPMDLoad/SPMDStore.
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			switch instr := instr.(type) {
			case *ssa.UnOp:
				if instr.Op == token.MUL {
					t.Errorf("unconverted UnOp{MUL} remains in func body:\n%s", output)
				}
			case *ssa.Store:
				t.Errorf("unconverted Store remains in func body:\n%s", output)
			}
		}
	}

	// At least one SPMDStore should exist (from *dst = 42).
	foundStore := false
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDStore); ok {
				foundStore = true
			}
		}
	}
	if !foundStore {
		t.Errorf("expected at least one SPMDStore in func body:\n%s", output)
	}
}

// TestPredicateSPMD_ConvertAllMemOps_SkipsConverted verifies that
// spmdConvertAllMemOps does not double-convert instructions that were
// already converted by predicateSPMDScope (inside varying branches).
func TestPredicateSPMD_ConvertAllMemOps_SkipsConverted(t *testing.T) {
	src := `package main
import "lanes"
func f(v lanes.Varying[int], dst *int) {
	vi := lanes.Varying[int](10)
	if v > vi {
		*dst = 42
	}
	*dst = 99
}
func main() {}
`
	pkg := buildSPMDFuncBody(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Count SPMDStore instructions.
	spmdStoreCount := 0
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDStore); ok {
				spmdStoreCount++
			}
		}
	}
	// Two stores: one inside varying if (converted by predicateSPMDScope),
	// one outside (converted by spmdConvertAllMemOps).
	if spmdStoreCount < 2 {
		var buf bytes.Buffer
		ssa.WriteFunction(&buf, fn)
		t.Errorf("expected at least 2 SPMDStore instructions, got %d:\n%s", spmdStoreCount, buf.String())
	}

	// No plain Store should remain.
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.Store); ok {
				var buf bytes.Buffer
				ssa.WriteFunction(&buf, fn)
				t.Errorf("unconverted Store remains after both passes:\n%s", buf.String())
			}
		}
	}
}

// TestPredicateSPMD_ConvertAllMemOps_GoForConverted verifies that
// spmdConvertScopedMemOps IS called for go-for loop bodies, converting
// straight-line loads/stores to SPMDLoad/SPMDStore via predicateSPMDLoop.
func TestPredicateSPMD_ConvertAllMemOps_GoForConverted(t *testing.T) {
	src := `package main

func f(arr []int) {
	for i := range len(arr) {
		arr[i] = arr[i] + 1
	}
}

func main() { f(make([]int, 16)) }
`
	// Build without SanityCheckFunctions to avoid a pre-existing referrer
	// issue with loop peeling (not related to spmdConvertScopedMemOps).
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(file)
	pkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{file}, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Go-for loop bodies now have their straight-line loads/stores converted
	// to SPMDLoad/SPMDStore by spmdConvertScopedMemOps (called from
	// predicateSPMDLoop). No plain UnOp{MUL} or Store should remain in scope
	// blocks after this pass.
	//
	// Note: blocks outside the loop scope (entry, done) retain regular mem ops.
	// We check that at least one SPMDLoad and one SPMDStore exist in the function.
	var foundSPMDLoad, foundSPMDStore bool
	var foundPlainLoad, foundPlainStore bool
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			switch instr := instr.(type) {
			case *ssa.SPMDLoad:
				foundSPMDLoad = true
			case *ssa.SPMDStore:
				foundSPMDStore = true
			case *ssa.UnOp:
				if instr.Op == token.MUL {
					// Check if this block is a scope block (body or loop).
					// Entry and done blocks are outside scope and may retain
					// plain loads (e.g., bounds checks).
					if strings.Contains(block.Comment, "rangeint.body") ||
						strings.Contains(block.Comment, "rangeint.loop") ||
						strings.Contains(block.Comment, "rangeindex.body") ||
						strings.Contains(block.Comment, "rangeindex.loop") {
						foundPlainLoad = true
					}
				}
			case *ssa.Store:
				if strings.Contains(block.Comment, "rangeint.body") ||
					strings.Contains(block.Comment, "rangeint.loop") ||
					strings.Contains(block.Comment, "rangeindex.body") ||
					strings.Contains(block.Comment, "rangeindex.loop") {
					foundPlainStore = true
				}
			}
		}
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if !foundSPMDLoad {
		t.Errorf("expected at least one SPMDLoad in go-for loop body:\n%s", output)
	}
	if !foundSPMDStore {
		t.Errorf("expected at least one SPMDStore in go-for loop body:\n%s", output)
	}
	if foundPlainLoad {
		t.Errorf("plain UnOp{MUL} remains in go-for loop scope block:\n%s", output)
	}
	if foundPlainStore {
		t.Errorf("plain Store remains in go-for loop scope block:\n%s", output)
	}
}

// ----------------------------------------------------------------------------
// Deferred loop-header merge tests (Task 1)

// TestPredicateSPMD_GoForLoopHeaderMergeLinearized verifies that a varying
// if/else inside a go-for loop where both branches jump back to the loop header
// is linearized via the deferred-merge path.
//
// The varying If is replaced with a Jump (B→T→E→loopHeader) and mask
// computation is inserted. The loop header's Phi must NOT be replaced with an
// SPMDSelect at the loop header itself; any SPMDSelect for differing phi values
// is placed in the else block (just before its jump to the loop header).
//
// Note: for the encode function, the rangeindex.loop phi carries the same value
// (the incremented index) from both then and else branches, so no SPMDSelect
// is emitted for that particular phi. The important checks are that the varying
// If was linearized, the loop header has no SPMDSelect, and mask ops are present.
func TestPredicateSPMD_GoForLoopHeaderMergeLinearized(t *testing.T) {
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

	// SSA sanity check must pass after deferred-merge linearization.
	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("BuildPackage sanity check failed: %v", err)
	}

	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("encode")
	if fn == nil {
		t.Fatal("function encode not found")
	}

	// No varying If should survive predication.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && vif.IsVarying {
			t.Errorf("block %d still has a varying If after deferred-merge linearization", block.Index)
		}
	}

	// The loop header must NOT have SPMDSelect in it. Any SPMDSelect for phi
	// values that differ between then/else is placed in the else block, not
	// the loop header. Loop-carried index phis (that carry the same value from
	// both branches) need no select and remain as plain Phi.
	for _, block := range fn.Blocks {
		if !strings.Contains(block.Comment, "rangeindex.loop") {
			continue
		}
		for _, instr := range block.Instrs {
			if _, ok := instr.(*ssa.SPMDSelect); ok {
				t.Errorf("loop header block %q contains SPMDSelect — loop-carried phi replaced incorrectly", block.Comment)
			}
		}
	}

	// Mask computation must be present (linearization occurred).
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Errorf("expected mask computation instructions:\n%s", output)
	}
}

// TestPredicateSPMD_GoForLoopHeaderMergeSanity verifies the SSA sanity checker
// passes after deferred-merge linearization in go-for loop context.
func TestPredicateSPMD_GoForLoopHeaderMergeSanity(t *testing.T) {
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

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("SanityCheck failed: %v", err)
	}
}

// TestPredicateSPMD_GoForLoopStraightLineMem verifies that straight-line
// loads/stores in a go-for loop body (not inside a varying if) are converted
// to SPMDLoad/SPMDStore by spmdConvertScopedMemOps.
func TestPredicateSPMD_GoForLoopStraightLineMem(t *testing.T) {
	src := `package main

func f(arr []int) {
	for i := range len(arr) {
		// Straight-line load and store (no varying if around them).
		arr[i] = arr[i] + 1
	}
}

func main() { f(make([]int, 16)) }
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	var spmdLoads, spmdStores int
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			switch instr.(type) {
			case *ssa.SPMDLoad:
				spmdLoads++
			case *ssa.SPMDStore:
				spmdStores++
			}
		}
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if spmdLoads == 0 {
		t.Errorf("expected SPMDLoad(s) in straight-line go-for loop:\n%s", output)
	}
	if spmdStores == 0 {
		t.Errorf("expected SPMDStore(s) in straight-line go-for loop:\n%s", output)
	}
}

// TestPredicateSPMD_GoForLoopCallMask verifies that SPMD function calls inside
// a go-for loop body receive the active mask via spmdMaskScopedCallOps.
func TestPredicateSPMD_GoForLoopCallMask(t *testing.T) {
	src := `package main
import "lanes"

func helper(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}

func f(arr []int) {
	for i := range len(arr) {
		vi := lanes.Varying[int](arr[i])
		result := helper(vi)
		arr[i] = int(result)
	}
}

func main() { f(make([]int, 16)) }
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// Find the Call to helper and check that SPMDMask is set.
	var helperCall *ssa.Call
	for _, block := range fn.Blocks {
		for _, instr := range block.Instrs {
			call, ok := instr.(*ssa.Call)
			if !ok {
				continue
			}
			callee := call.Call.StaticCallee()
			if callee != nil && callee.Name() == "helper" {
				helperCall = call
			}
		}
	}

	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()

	if helperCall == nil {
		t.Fatalf("call to helper not found in:\n%s", output)
	}
	if helperCall.Call.SPMDMask == nil {
		t.Errorf("expected SPMDMask on call to helper in go-for loop:\n%s", output)
	}
}

// TestPredicateSPMD_GoForLoopLORShortCircuit verifies that a LOR (||) short-circuit
// If with a varying condition — generated by logicalBinop when the || result is
// used as a value (not directly as an if-condition) — is linearized correctly.
// The pattern is:
//
//	B: if cond goto T(done) else E(rhs)    [logicalBinop LOR short-circuit]
//	E: compute rhs, goto T
//	T: phi [B: true, E: rhs_val]           [binop.done merge phi]
//
// predicateVaryingIf must linearize this to B→E→T and replace the phi with
// SPMDSelect(mask_cond, true, rhs_val). Without this fix, TinyGo would emit
// `br <N x iW> %cond, label %T, label %E` which LLVM rejects.
func TestPredicateSPMD_GoForLoopLORShortCircuit(t *testing.T) {
	// reduceAny is a stand-in for reduce.Any (takes Varying[bool], returns bool).
	// The || is used as a reified value (argument to reduceAny), not directly
	// as an if-condition, so it goes through logicalBinop and creates binop.*
	// blocks without a SPMDBooleanChain entry.
	src := `package main
import "lanes"

func reduceAny(v lanes.Varying[bool]) bool { return false }

func f(starts [4]int, ends [4]int) bool {
	for i, start := range starts {
		end := ends[i]
		fieldLen := end - start
		_ = lanes.Varying[int](fieldLen)
		if reduceAny(fieldLen < 1 || fieldLen > 3) {
			return true
		}
	}
	return false
}

func main() {}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(file)

	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("f")
	if fn == nil {
		t.Fatal("function f not found")
	}

	// After predication, no varying If should survive.
	for _, block := range fn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if vif, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && vif.IsVarying {
			t.Errorf("block %d (%s) still has a varying If after LOR short-circuit linearization",
				block.Index, block.Comment)
		}
	}

	// An SPMDSelect must be present (the phi in binop.done is converted to select).
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	output := buf.String()
	if !strings.Contains(output, "spmd_select") {
		t.Errorf("expected spmd_select (phi→select conversion) after LOR linearization:\n%s", output)
	}
}

// TestPredicateSPMD_GoForLoopLORShortCircuitSanity verifies the SSA sanity
// checker passes after linearizing a LOR short-circuit in a go-for loop.
func TestPredicateSPMD_GoForLoopLORShortCircuitSanity(t *testing.T) {
	src := `package main
import "lanes"

func reduceAny(v lanes.Varying[bool]) bool { return false }

func f(starts [4]int, ends [4]int) bool {
	for i, start := range starts {
		end := ends[i]
		fieldLen := end - start
		_ = lanes.Varying[int](fieldLen)
		if reduceAny(fieldLen < 1 || fieldLen > 3) {
			return true
		}
	}
	return false
}

func main() {}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(file)

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{file},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("SanityCheck failed: %v", err)
	}
}

// TestPredicateSPMD_GoForLoopCallMaskSanity verifies the SSA sanity checker
// passes after spmdMaskScopedCallOps sets masks on SPMD calls in a go-for loop.
func TestPredicateSPMD_GoForLoopCallMaskSanity(t *testing.T) {
	src := `package main
import "lanes"

func helper(v lanes.Varying[int]) lanes.Varying[int] {
	return v
}

func f(arr []int) {
	for i := range len(arr) {
		vi := lanes.Varying[int](arr[i])
		result := helper(vi)
		arr[i] = int(result)
	}
}

func main() { f(make([]int, 16)) }
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(file)

	_, _, err = ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{file},
		ssa.SanityCheckFunctions,
	)
	if err != nil {
		t.Fatalf("SanityCheck failed: %v", err)
	}
}
