package spmd

import (
	"go/types"
	"testing"
)

func TestMaskTypeImplementsType(t *testing.T) {
	var _ types.Type = MaskInstance
}

func TestMaskTypeUnderlying(t *testing.T) {
	if MaskInstance.Underlying() != MaskInstance {
		t.Error("MaskType.Underlying() should return itself")
	}
}

func TestMaskTypeString(t *testing.T) {
	if s := MaskInstance.String(); s != "mask" {
		t.Errorf("MaskType.String() = %q, want %q", s, "mask")
	}
}

func TestIsMask(t *testing.T) {
	if !IsMask(MaskInstance) {
		t.Error("IsMask(MaskInstance) = false, want true")
	}
	if IsMask(types.Typ[types.Bool]) {
		t.Error("IsMask(Bool) = true, want false")
	}
}

func TestIsVaryingMask(t *testing.T) {
	vm := NewVaryingMask()
	if !IsVaryingMask(vm) {
		t.Error("IsVaryingMask(NewVaryingMask()) = false, want true")
	}
	// Varying[int32] is not a varying mask
	vi := types.NewVarying(types.Typ[types.Int32])
	if IsVaryingMask(vi) {
		t.Error("IsVaryingMask(Varying[int32]) = true, want false")
	}
	// Bare MaskType is not a varying mask (not wrapped in SPMDType)
	if IsVaryingMask(MaskInstance) {
		t.Error("IsVaryingMask(MaskInstance) = true, want false")
	}
}

func TestNewVaryingMask(t *testing.T) {
	vm := NewVaryingMask()
	if !vm.IsVarying() {
		t.Error("NewVaryingMask().IsVarying() = false, want true")
	}
	if !IsMask(vm.Elem()) {
		t.Error("NewVaryingMask().Elem() is not MaskType")
	}
}

func TestMaskSingleton(t *testing.T) {
	// Two calls to NewVaryingMask use the same MaskInstance
	vm1 := NewVaryingMask()
	vm2 := NewVaryingMask()
	if vm1.Elem() != vm2.Elem() {
		t.Error("MaskInstance is not a singleton")
	}
}
