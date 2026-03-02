package ssa_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// setSPMDOnRange walks the AST and sets IsSpmd=true, LaneCount=4
// on the first RangeStmt found.
func setSPMDOnRange(f *ast.File) {
	ast.Inspect(f, func(n ast.Node) bool {
		if rs, ok := n.(*ast.RangeStmt); ok {
			rs.IsSpmd = true
			rs.LaneCount = 4
			return false
		}
		return true
	})
}

func buildSSAWithSPMD(t *testing.T, src string) *ssa.Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	setSPMDOnRange(f)

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

func TestSPMDLoopInfo_RangeInt(t *testing.T) {
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

	if len(mainFn.SPMDLoops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(mainFn.SPMDLoops))
	}

	info := mainFn.SPMDLoops[0]

	// Block pointers must be non-nil
	if info.EntryBlock == nil {
		t.Error("EntryBlock is nil")
	}
	if info.BodyBlock == nil {
		t.Error("BodyBlock is nil")
	}
	if info.DoneBlock == nil {
		t.Error("DoneBlock is nil")
	}

	// LaneCount is set by the type checker based on the element type width;
	// just verify it is positive.
	if info.LaneCount <= 0 {
		t.Errorf("LaneCount = %d, want > 0", info.LaneCount)
	}
	if info.IsRangeIndex {
		t.Error("IsRangeIndex should be false for range-over-int")
	}
	if info.BoundValue == nil {
		t.Error("BoundValue is nil")
	}
}

func TestSPMDLoopInfo_RangeIndexed(t *testing.T) {
	src := `package main

func main() {
	data := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := range data {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if mainFn == nil {
		t.Fatal("main function not found")
	}

	if len(mainFn.SPMDLoops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(mainFn.SPMDLoops))
	}

	info := mainFn.SPMDLoops[0]

	if info.EntryBlock == nil {
		t.Error("EntryBlock is nil")
	}
	if info.BodyBlock == nil {
		t.Error("BodyBlock is nil")
	}
	if info.LoopBlock == nil {
		t.Error("LoopBlock is nil")
	}
	if info.DoneBlock == nil {
		t.Error("DoneBlock is nil")
	}
	if !info.IsRangeIndex {
		t.Error("IsRangeIndex should be true for range-over-slice")
	}
	// LaneCount is set by the type checker based on element type width;
	// just verify it is positive.
	if info.LaneCount <= 0 {
		t.Errorf("LaneCount = %d, want > 0", info.LaneCount)
	}
}
