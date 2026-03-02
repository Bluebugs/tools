package ssa_test

import (
	"bytes"
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

func TestSPMDLoopInfo_IterPhi_RangeInt(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if len(mainFn.SPMDLoops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(mainFn.SPMDLoops))
	}

	info := mainFn.SPMDLoops[0]

	if info.IterPhi == nil {
		t.Fatal("IterPhi is nil after lift")
	}
	if info.IterPhi.Comment != "rangeint.iter" {
		t.Errorf("IterPhi.Comment = %q, want %q", info.IterPhi.Comment, "rangeint.iter")
	}

	if info.IncrBinOp == nil {
		t.Fatal("IncrBinOp is nil after lift")
	}

	// MergedBodyLoop: for rangeInt, body and loop are typically fused.
	if info.MergedBodyLoop {
		if info.BodyBlock != info.LoopBlock {
			t.Error("MergedBodyLoop=true but BodyBlock != LoopBlock")
		}
	}
}

func TestSPMDLoopInfo_Accumulators(t *testing.T) {
	// Use a uniform accumulator (sum += 1) to satisfy the SPMD type checker:
	// inside a go-for loop, the iteration variable i is varying, so assigning
	// it to a uniform sum would be a type error. Accumulating a uniform
	// constant is valid and still produces a loop-carried phi after lift.
	src := `package main

func main() {
	sum := 0
	for i := range 16 {
		_ = i
		sum += 1
	}
	_ = sum
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if len(mainFn.SPMDLoops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(mainFn.SPMDLoops))
	}

	info := mainFn.SPMDLoops[0]

	// sum is a loop-carried accumulator.
	if len(info.Accumulators) < 1 {
		t.Fatalf("expected at least 1 accumulator, got %d", len(info.Accumulators))
	}

	acc := info.Accumulators[0]
	if acc.Phi == nil {
		t.Error("Accumulator.Phi is nil")
	}
	if acc.InitValue == nil {
		t.Error("Accumulator.InitValue is nil")
	}
	if acc.BackValue == nil {
		t.Error("Accumulator.BackValue is nil")
	}
}

func TestSPMDLoopInfo_IterPhi_RangeIndexed(t *testing.T) {
	src := `package main

func main() {
	data := []int{1, 2, 3, 4}
	for i := range data {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if len(mainFn.SPMDLoops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(mainFn.SPMDLoops))
	}

	info := mainFn.SPMDLoops[0]

	if info.IterPhi == nil {
		t.Fatal("IterPhi is nil after lift for rangeIndexed")
	}
	if info.IterPhi.Comment != "rangeindex" {
		t.Errorf("IterPhi.Comment = %q, want %q", info.IterPhi.Comment, "rangeindex")
	}
	if info.IsRangeIndex != true {
		t.Error("IsRangeIndex should be true")
	}
	// rangeIndexed blocks should NOT be merged.
	if info.MergedBodyLoop {
		t.Error("MergedBodyLoop should be false for rangeIndexed")
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

func TestSPMDLoopInfo_String(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()

	if !strings.Contains(output, "SPMDLoop") {
		t.Errorf("WriteTo output does not contain SPMDLoop info:\n%s", output)
	}
}

func TestSPMDLoopInfo_NonSPMD(t *testing.T) {
	src := `package main

func main() {
	sum := 0
	for i := range 16 {
		sum += i
	}
	_ = sum
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Deliberately NOT setting IsSpmd

	mode := ssa.SanityCheckFunctions
	pkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: importer.Default()},
		fset, types.NewPackage("main", ""), []*ast.File{f}, mode,
	)
	if err != nil {
		t.Fatal(err)
	}

	mainFn := pkg.Func("main")
	if len(mainFn.SPMDLoops) != 0 {
		t.Errorf("expected 0 SPMD loops for non-SPMD range, got %d", len(mainFn.SPMDLoops))
	}
}

func TestSPMDLoopInfo_MultipleLoops(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		_ = i
	}
	for j := range 32 {
		_ = j
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Set IsSpmd on ALL RangeStmts
	ast.Inspect(f, func(n ast.Node) bool {
		if rs, ok := n.(*ast.RangeStmt); ok {
			rs.IsSpmd = true
			rs.LaneCount = 4
		}
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

	mainFn := pkg.Func("main")
	if len(mainFn.SPMDLoops) != 2 {
		t.Fatalf("expected 2 SPMD loops, got %d", len(mainFn.SPMDLoops))
	}

	for i, info := range mainFn.SPMDLoops {
		if info.IterPhi == nil {
			t.Errorf("loop %d: IterPhi is nil", i)
		}
		if info.EntryBlock == nil {
			t.Errorf("loop %d: EntryBlock is nil", i)
		}
		if info.DoneBlock == nil {
			t.Errorf("loop %d: DoneBlock is nil", i)
		}
	}
}

func TestSPMDLoopInfo_MultipleAccumulators(t *testing.T) {
	src := `package main

func main() {
	sum := 0
	count := 0
	for i := range 16 {
		_ = i
		sum += 1
		count += 1
	}
	_ = sum
	_ = count
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")
	if len(mainFn.SPMDLoops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(mainFn.SPMDLoops))
	}

	info := mainFn.SPMDLoops[0]
	if len(info.Accumulators) < 2 {
		t.Fatalf("expected at least 2 accumulators (sum, count), got %d", len(info.Accumulators))
	}
}
