// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// buildSSAWithSPMDLaneCounts is like buildSSAWithSPMD but marks successive
// RangeStmts with the provided lane counts. If there are more RangeStmts
// than lane counts, the remaining ranges are left unmodified (not SPMD).
// If a lane count entry is 0, the corresponding range is skipped (not SPMD).
func buildSSAWithSPMDLaneCounts(t *testing.T, src string, laneCounts []int64) *ssa.Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	idx := 0
	ast.Inspect(f, func(n ast.Node) bool {
		if idx >= len(laneCounts) {
			return false
		}
		rs, ok := n.(*ast.RangeStmt)
		if !ok {
			return true
		}
		if lc := laneCounts[idx]; lc > 0 {
			rs.IsSpmd = true
			rs.LaneCount = lc
		}
		idx++
		return true
	})
	mode := ssa.SanityCheckFunctions
	pkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f}, mode,
	)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

// hasSPMDVariantPrefix returns true if the name is "sum.spmd" followed by digits.
func hasSPMDVariantPrefix(name, funcName string) bool {
	prefix := funcName + ".spmd"
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	suffix := name[len(prefix):]
	if len(suffix) == 0 {
		return false
	}
	for _, ch := range suffix {
		if ch < '0' || ch > '9' {
			return false
		}
	}
	return true
}

// TestSPMDBlockLaneCountSet verifies that the SPMD predication pass
// annotates every in-scope block of an SPMD loop with the loop's
// canonical lane count via *ssa.BasicBlock.SPMDLaneCount.
func TestSPMDBlockLaneCountSet(t *testing.T) {
	pkg := buildSSAWithSPMD(t, accumulateSrc)
	fn := pkg.Func("accumulate")
	if fn == nil {
		t.Fatal("accumulate function not found in SSA")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("accumulate has no SPMD loops; buildSSAWithSPMD should mark one")
	}
	loop := fn.SPMDLoops[0]
	if loop.LaneCount <= 0 {
		t.Fatalf("loop.LaneCount = %d; expected > 0", loop.LaneCount)
	}

	// At least one block in the function must have SPMDLaneCount = loop.LaneCount.
	var annotated int
	for _, bb := range fn.Blocks {
		if bb.SPMDLaneCount == loop.LaneCount {
			annotated++
		}
	}
	if annotated == 0 {
		t.Fatalf("no blocks have SPMDLaneCount = %d; predication pass did not annotate", loop.LaneCount)
	}

	// Suppress unused-import warning when ssa.* not directly referenced.
	_ = ssa.SPMDLoopInfo{}
}

// TestSPMDForwardPropagationEntryBlock verifies that an entry-block alloca
// consumed by SPMD ops in an annotated block causes the alloca's containing
// block to receive an annotation matching the consumer's lane count.
//
// The fixture: main() declares `var acc lanes.Varying[int]` in its entry
// block, then uses it inside a go for that has SPMDLaneCount > 0. Forward
// propagation should walk the alloca's referrers, find the in-loop
// SPMDStore/SPMDLoad, and annotate main()'s entry block with that loop's
// lane count.
func TestSPMDForwardPropagationEntryBlock(t *testing.T) {
	// Use the same fixture pattern as accumulateSrc: a Varying[int] accumulator
	// declared in the entry block, read and written inside the SPMD loop body.
	// buildSSAWithSPMD marks the first RangeStmt as IsSpmd=true (LaneCount=4),
	// which causes spmdConvertLoopOps to annotate the loop body blocks with
	// SPMDLaneCount=4. The entry block (which owns the alloca) starts at 0.
	// spmdPropagateBlockLaneCount must walk the alloca's referrers, find the
	// Store inside the annotated loop body, and propagate SPMDLaneCount=4 to
	// the entry block.
	src := `package main

import "lanes"

func main() {
	var acc lanes.Varying[int]
	for i := range 16 {
		_ = i
		acc = acc
	}
	_ = acc
}
`
	pkg := buildSSAWithSPMD(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found in SSA")
	}
	if len(fn.Blocks) == 0 {
		t.Fatal("main has no blocks")
	}
	entry := fn.Blocks[0]
	if entry.SPMDLaneCount == 0 {
		t.Fatalf("entry block SPMDLaneCount = 0; expected forward-propagation to annotate it from the SPMD consumer")
	}
	if len(fn.SPMDLoops) > 0 && entry.SPMDLaneCount != fn.SPMDLoops[0].LaneCount {
		t.Fatalf("entry block SPMDLaneCount = %d; expected %d (loop's lane count)",
			entry.SPMDLaneCount, fn.SPMDLoops[0].LaneCount)
	}
}

// TestSPMDSpecializationCloneVariants verifies that calling an SPMD function
// from two different lane counts produces two SSA function variants
// (e.g., sum.spmd2 and sum.spmd4) with each variant's blocks annotated
// to the variant's lane count.
func TestSPMDSpecializationCloneVariants(t *testing.T) {
	t.Skip("SPMD v4.1 deferred: specialization pass disabled pending TinyGo cloned-variant emission fix")
	src := `package main

import "lanes"

func sum(v lanes.Varying[int]) int {
	var s int
	_ = v
	return s
}

var floats = []float64{1, 2, 3, 4}
var ints = []int32{1, 2, 3, 4, 5, 6, 7, 8}

func main() {
	var a lanes.Varying[int]
	for i := range len(floats) { _ = i; _ = a }
	_ = sum(a)

	var b lanes.Varying[int]
	for i := range len(ints) { _ = i; _ = b }
	_ = sum(b)
}
`
	// Mark first loop with LaneCount=2, second with LaneCount=4.
	pkg := buildSSAWithSPMDLaneCounts(t, src, []int64{2, 4})
	var variants []*ssa.Function
	for name, mem := range pkg.Members {
		if fn, ok := mem.(*ssa.Function); ok {
			if hasSPMDVariantPrefix(name, "sum") {
				variants = append(variants, fn)
			}
		}
	}
	if len(variants) < 2 {
		t.Fatalf("expected >=2 sum.spmd<N> variants, got %d", len(variants))
	}
	for _, fn := range variants {
		hasPos := false
		for _, bb := range fn.Blocks {
			if bb.SPMDLaneCount > 0 {
				hasPos = true
				break
			}
		}
		if !hasPos {
			t.Errorf("variant %s has no annotated blocks", fn.Name())
		}
	}
}

// TestSPMDSpecializationRewriteCallSites verifies that after specialization
// the call sites in annotated blocks point to the specialized variant,
// not the original function.
func TestSPMDSpecializationRewriteCallSites(t *testing.T) {
	t.Skip("SPMD v4.1 deferred: specialization pass disabled pending TinyGo cloned-variant emission fix")
	src := `package main

import "lanes"

func sum(v lanes.Varying[int]) int { return 0 }

var floats = []float64{1, 2}

func main() {
	var a lanes.Varying[int]
	for i := range len(floats) { _ = i; _ = a }
	_ = sum(a)
}
`
	pkg := buildSSAWithSPMD(t, src)
	main := pkg.Func("main")
	if main == nil {
		t.Fatal("main not found")
	}
	var found bool
	for _, bb := range main.Blocks {
		for _, instr := range bb.Instrs {
			if call, ok := instr.(*ssa.Call); ok {
				if call.Call.Value != nil {
					name := call.Call.Value.Name()
					if hasSPMDVariantPrefix(name, "sum") {
						found = true
					}
				}
			}
		}
	}
	if !found {
		t.Fatal("no call to sum.spmd<N> variant in main; specialization did not rewrite call sites")
	}
}

// TestSPMDSpecializationSingleLaneShortcut verifies that an SPMD function
// called from exactly one lane count is renamed in place (single variant)
// rather than cloned.
func TestSPMDSpecializationSingleLaneShortcut(t *testing.T) {
	t.Skip("SPMD v4.1 deferred: specialization pass disabled pending TinyGo cloned-variant emission fix")
	src := `package main

import "lanes"

func sum(v lanes.Varying[int]) int { return 0 }

var data = []float64{1, 2, 3, 4}

func main() {
	var a lanes.Varying[int]
	for i := range len(data) { _ = i; _ = a }
	_ = sum(a)
}
`
	pkg := buildSSAWithSPMD(t, src)
	var sumCount, variantCount int
	for name := range pkg.Members {
		if name == "sum" {
			sumCount++
		}
		if hasSPMDVariantPrefix(name, "sum") {
			variantCount++
		}
	}
	total := sumCount + variantCount
	if total != 1 {
		t.Fatalf("expected exactly 1 of sum / sum.spmdN, got sum=%d variants=%d", sumCount, variantCount)
	}
}
