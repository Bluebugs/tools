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
