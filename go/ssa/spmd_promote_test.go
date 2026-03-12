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
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
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
