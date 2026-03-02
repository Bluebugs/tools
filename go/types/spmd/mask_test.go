// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spmd_test

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/types/spmd"
)

func TestMaskTypeImplementsType(t *testing.T) {
	var _ types.Type = spmd.MaskInstance
}

func TestMaskTypeUnderlying(t *testing.T) {
	if spmd.MaskInstance.Underlying() != spmd.MaskInstance {
		t.Error("MaskType.Underlying() should return itself")
	}
}

func TestMaskTypeString(t *testing.T) {
	if s := spmd.MaskInstance.String(); s != "mask" {
		t.Errorf("MaskType.String() = %q, want %q", s, "mask")
	}
}

func TestIsMask(t *testing.T) {
	if !spmd.IsMask(spmd.MaskInstance) {
		t.Error("IsMask(MaskInstance) = false, want true")
	}
	if spmd.IsMask(types.Typ[types.Bool]) {
		t.Error("IsMask(Bool) = true, want false")
	}
}

func TestIsVaryingMask(t *testing.T) {
	vm := spmd.NewVaryingMask()
	if !spmd.IsVaryingMask(vm) {
		t.Error("IsVaryingMask(NewVaryingMask()) = false, want true")
	}
	// Varying[int32] is not a varying mask
	vi := types.NewVarying(types.Typ[types.Int32])
	if spmd.IsVaryingMask(vi) {
		t.Error("IsVaryingMask(Varying[int32]) = true, want false")
	}
	// Bare MaskType is not a varying mask (not wrapped in SPMDType)
	if spmd.IsVaryingMask(spmd.MaskInstance) {
		t.Error("IsVaryingMask(MaskInstance) = true, want false")
	}
}

func TestNewVaryingMask(t *testing.T) {
	vm := spmd.NewVaryingMask()
	if !vm.IsVarying() {
		t.Error("NewVaryingMask().IsVarying() = false, want true")
	}
	if !spmd.IsMask(vm.Elem()) {
		t.Error("NewVaryingMask().Elem() is not MaskType")
	}
}

func TestMaskSingleton(t *testing.T) {
	// Two calls to NewVaryingMask use the same MaskInstance
	vm1 := spmd.NewVaryingMask()
	vm2 := spmd.NewVaryingMask()
	if vm1.Elem() != vm2.Elem() {
		t.Error("MaskInstance is not a singleton")
	}
}
