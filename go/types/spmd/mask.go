// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package spmd defines types for SPMD (Single Program Multiple Data)
// support in the SSA pipeline. These types are internal to the
// compiler toolchain and are not exposed to user code.
package spmd

import "go/types"

// MaskType represents an opaque SPMD execution mask.
// It is internal to the SSA/compiler pipeline and never appears in user code.
// The concrete representation (bit width, format) is architecture-dependent.
//
// MaskType implements [go/types.Type] so it can be used as the element type
// inside [types.SPMDType]: SPMDType{VaryingQualifier, MaskType{}}.
//
// Constraints:
//   - Cannot be stored in memory (registers and mask stack only)
//   - Convertible to/from Varying[bool] via explicit SSA operations
type MaskType struct{}

// MaskInstance is the singleton MaskType value.
var MaskInstance = &MaskType{}

// Underlying returns the mask type itself (it has no underlying type).
func (t *MaskType) Underlying() types.Type { return t }

// String returns "mask".
func (t *MaskType) String() string { return "mask" }

// IsMask reports whether t is a *MaskType.
func IsMask(t types.Type) bool {
	_, ok := t.(*MaskType)
	return ok
}

// IsVaryingMask reports whether t is SPMDType{Varying, MaskType{}}.
func IsVaryingMask(t types.Type) bool {
	if st, ok := t.(*types.SPMDType); ok && st.IsVarying() {
		return IsMask(st.Elem())
	}
	return false
}

// NewVaryingMask returns SPMDType{VaryingQualifier, MaskType{}}.
func NewVaryingMask() *types.SPMDType {
	return types.NewVarying(MaskInstance)
}
