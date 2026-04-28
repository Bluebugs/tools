// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Test-only re-exports.

// IsLanesVaryingType exposes the file-private isLanesVaryingType helper to
// the external ssa_test package. Production callers use the lowercase
// symbol within package ssa.
var IsLanesVaryingType = isLanesVaryingType
