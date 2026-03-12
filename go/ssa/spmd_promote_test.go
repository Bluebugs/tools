// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

// Tests for SPMDVectorFromMemory instruction.
//
// These tests verify:
//  1. Type() returns *types.SPMDType with correct elem type
//  2. Operands() returns 2 pointers (&v.Ptr, &v.Len)
//  3. Mutating via Operands() pointer updates the field
//  4. String() starts with "spmd_vector_from_memory" and contains lane count
//  5. Pos() returns token.NoPos for default
//  6. Referrers() returns non-nil (value-producing instruction)
//  7. Interface compliance (compile-time checks)

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// Compile-time interface compliance checks.
var _ ssa.Value = (*ssa.SPMDVectorFromMemory)(nil)
var _ ssa.Instruction = (*ssa.SPMDVectorFromMemory)(nil)

// TestSPMDVectorFromMemory_Type_Byte verifies Type() returns Varying[byte] for byte elem.
func TestSPMDVectorFromMemory_Type_Byte(t *testing.T) {
	byteType := types.Typ[types.Byte]
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptrConst(),
		Len:      intConst(16),
		ElemType: byteType,
		Lanes:    16,
	}

	got := v.Type()
	want := types.NewVarying(byteType)
	if !types.Identical(got, want) {
		t.Errorf("SPMDVectorFromMemory.Type() = %s, want %s", got, want)
	}
}

// TestSPMDVectorFromMemory_Type_Int32 verifies Type() returns Varying[int32] for int32 elem.
func TestSPMDVectorFromMemory_Type_Int32(t *testing.T) {
	int32Type := types.Typ[types.Int32]
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptrConst(),
		Len:      intConst(4),
		ElemType: int32Type,
		Lanes:    4,
	}

	got := v.Type()
	want := types.NewVarying(int32Type)
	if !types.Identical(got, want) {
		t.Errorf("SPMDVectorFromMemory.Type() = %s, want %s", got, want)
	}
}

// TestSPMDVectorFromMemory_Operands verifies Operands returns 2 pointers: &Ptr and &Len.
func TestSPMDVectorFromMemory_Operands(t *testing.T) {
	ptr := ptrConst()
	length := intConst(4)
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptr,
		Len:      length,
		ElemType: types.Typ[types.Int32],
		Lanes:    4,
	}

	rands := v.Operands(nil)
	if len(rands) != 2 {
		t.Fatalf("Operands() returned %d pointers, want 2", len(rands))
	}
	if *rands[0] != ssa.Value(ptr) {
		t.Errorf("rands[0] does not point to Ptr")
	}
	if *rands[1] != ssa.Value(length) {
		t.Errorf("rands[1] does not point to Len")
	}
}

// TestSPMDVectorFromMemory_OperandsMutate verifies mutating via Operands() pointer updates the field.
func TestSPMDVectorFromMemory_OperandsMutate(t *testing.T) {
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptrConst(),
		Len:      intConst(4),
		ElemType: types.Typ[types.Int32],
		Lanes:    4,
	}

	newLen := intConst(99)
	rands := v.Operands(nil)
	// rands[1] is &v.Len; mutate it
	*rands[1] = newLen

	if v.Len != ssa.Value(newLen) {
		t.Errorf("after mutating rands[1], v.Len = %v, want newLen", v.Len)
	}
}

// TestSPMDVectorFromMemory_String verifies the string representation.
func TestSPMDVectorFromMemory_String(t *testing.T) {
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptrConst(),
		Len:      intConst(4),
		ElemType: types.Typ[types.Int32],
		Lanes:    4,
	}

	s := v.String()
	if !strings.HasPrefix(s, "spmd_vector_from_memory") {
		t.Errorf("String() = %q, want prefix %q", s, "spmd_vector_from_memory")
	}
	if !strings.Contains(s, "4") {
		t.Errorf("String() = %q, expected to contain lane count %q", s, "4")
	}
}

// TestSPMDVectorFromMemory_Pos verifies Pos() returns token.NoPos for default construction.
func TestSPMDVectorFromMemory_Pos(t *testing.T) {
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptrConst(),
		Len:      intConst(4),
		ElemType: types.Typ[types.Int32],
		Lanes:    4,
	}

	if got := v.Pos(); got != token.NoPos {
		t.Errorf("Pos() = %v, want token.NoPos", got)
	}
}

// TestSPMDVectorFromMemory_Referrers verifies Referrers() returns non-nil for a value-producing instruction.
func TestSPMDVectorFromMemory_Referrers(t *testing.T) {
	v := &ssa.SPMDVectorFromMemory{
		Ptr:      ptrConst(),
		Len:      intConst(4),
		ElemType: types.Typ[types.Int32],
		Lanes:    4,
	}

	if v.Referrers() == nil {
		t.Error("Referrers() returned nil, want non-nil slice pointer")
	}
}

// buildSPMDProgram builds a package with SPMD loops at the given lane count.
// All RangeStmts in the source are marked as SPMD with laneCount.
func buildSPMDProgram(t *testing.T, src string, laneCount int) *ssa.Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	ast.Inspect(f, func(n ast.Node) bool {
		if rs, ok := n.(*ast.RangeStmt); ok {
			rs.IsSpmd = true
			rs.LaneCount = int64(laneCount)
		}
		return true
	})
	pkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f}, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

// TestPromoteSPMDArrays_ByteArray verifies a [16]byte array inside a
// 16-lane go for loop is promoted (Alloc removed).
func TestPromoteSPMDArrays_ByteArray(t *testing.T) {
	src := `package main

func main() {
	var arr [16]byte
	for i := range 16 {
		arr[i] = byte(i)
	}
}
`
	pkg := buildSPMDProgram(t, src, 16)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main not found")
	}

	// After promotion, there should be no Alloc for [16]byte in the function.
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if at, ok := alloc.Type().Underlying().(*types.Pointer); ok {
					if _, ok := at.Elem().Underlying().(*types.Array); ok {
						t.Errorf("found unpromoted array Alloc: %s", alloc)
					}
				}
			}
		}
	}
}

// TestPromoteSPMDArrays_Ineligible_Escape verifies arrays passed to calls
// are NOT promoted.
func TestPromoteSPMDArrays_Ineligible_Escape(t *testing.T) {
	src := `package main

func use(p *[16]byte) {}

func main() {
	var arr [16]byte
	for i := range 16 {
		arr[i] = byte(i)
	}
	use(&arr)
}
`
	pkg := buildSPMDProgram(t, src, 16)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main not found")
	}

	found := false
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.Alloc); ok {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected Alloc to remain (ineligible for promotion)")
	}
}

// TestPromoteSPMDArrays_Ineligible_LaneMismatch verifies [8]byte in a
// 16-lane loop is NOT promoted.
func TestPromoteSPMDArrays_Ineligible_LaneMismatch(t *testing.T) {
	src := `package main

func main() {
	var arr [8]byte
	for i := range 16 {
		if i < 8 {
			arr[i] = byte(i)
		}
	}
}
`
	pkg := buildSPMDProgram(t, src, 16)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main not found")
	}

	found := false
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.Alloc); ok {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected Alloc to remain (lane count mismatch)")
	}
}

// TestPromoteSPMDArrays_EmitsVectorFromMemory verifies that the copy+read
// pattern (copy(input[:], s) + loop reads) is promoted to SPMDVectorFromMemory.
func TestPromoteSPMDArrays_EmitsVectorFromMemory(t *testing.T) {
	src := `package main

func process(s string) {
	var input [16]byte
	copy(input[:], s)
	for i := range 16 {
		_ = input[i] - 48
	}
}

func main() {
	process("hello")
}
`
	pkg := buildSPMDProgram(t, src, 16)
	fn := pkg.Func("process")
	if fn == nil {
		t.Fatal("process not found")
	}

	// After promotion, there should be an SPMDVectorFromMemory instruction.
	found := false
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.SPMDVectorFromMemory); ok {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected SPMDVectorFromMemory instruction after promotion")
		fn.WriteTo(os.Stderr)
	}

	// No Alloc for [16]byte should remain.
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if at, ok := alloc.Type().Underlying().(*types.Pointer); ok {
					if _, ok := at.Elem().Underlying().(*types.Array); ok {
						t.Errorf("found unpromoted array Alloc: %s", alloc)
					}
				}
			}
		}
	}
}

// TestPromoteSPMDArrays_CopyRangeInput verifies that the copy+rangeindex
// pattern (copy(input[:], s) followed by a rangeindex loop reading input[i])
// is promoted to SPMDVectorFromMemory. This matches the pattern in the IPv4
// parser's parseIPv4 function where `go for i, c := range input` reads a
// [16]byte buffer filled by copy(input[:], s).
func TestPromoteSPMDArrays_CopyRangeInput(t *testing.T) {
	src := `package main

import "lanes"

func process(s string) lanes.Varying[byte] {
	var input [16]byte
	copy(input[:], s)
	var result lanes.Varying[byte]
	for i, c := range input {
		_ = i
		result = c
	}
	return result
}

func main() {
	process("hello")
}
`
	pkg := buildSPMDProgram(t, src, 16)
	fn := pkg.Func("process")
	if fn == nil {
		t.Fatal("process not found")
	}

	// After promotion, there should be an SPMDVectorFromMemory and no array Alloc.
	foundVFM := false
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.SPMDVectorFromMemory); ok {
				foundVFM = true
			}
			if alloc, ok := instr.(*ssa.Alloc); ok {
				if at, ok := alloc.Type().Underlying().(*types.Pointer); ok {
					if _, ok := at.Elem().Underlying().(*types.Array); ok {
						t.Errorf("found unpromoted array Alloc: %s", alloc)
					}
				}
			}
		}
	}
	if !foundVFM {
		t.Error("expected SPMDVectorFromMemory instruction after promotion")
		fn.WriteTo(os.Stderr)
	}
}

// TestPromoteSPMDArrays_Ineligible_UniformIndex verifies arrays accessed
// with a uniform (non-IterPhi) index are NOT promoted.
func TestPromoteSPMDArrays_Ineligible_UniformIndex(t *testing.T) {
	src := `package main

func main() {
	var arr [16]byte
	for i := range 16 {
		arr[0] = byte(i) // uniform index 0, not IterPhi
	}
}
`
	pkg := buildSPMDProgram(t, src, 16)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main not found")
	}

	found := false
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.Alloc); ok {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected Alloc to remain (uniform index)")
	}
}
