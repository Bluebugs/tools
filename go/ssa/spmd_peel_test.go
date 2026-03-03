// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Tests for spmdCloneBlock in spmd_peel.go.
//
// Uses the internal (package ssa) test package to access newBasicBlock,
// which is not exported. We cannot import ssautil here because ssautil
// imports ssa, which would create an import cycle.

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"
)

// buildSSAForPeelTest builds a fully-built SSA Package from src, marking the
// first RangeStmt as SPMD (IsSpmd=true, LaneCount=4). It replicates the logic
// of ssautil.BuildPackage without importing ssautil to avoid an import cycle.
func buildSSAForPeelTest(t *testing.T, src string) *Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Mark the first RangeStmt as SPMD.
	ast.Inspect(f, func(n ast.Node) bool {
		if rs, ok := n.(*ast.RangeStmt); ok {
			rs.IsSpmd = true
			rs.LaneCount = 4
			return false
		}
		return true
	})

	// Type-check.
	tc := &types.Config{Importer: importer.Default()}
	typPkg := types.NewPackage("main", "")
	info := &types.Info{
		Types:        make(map[ast.Expr]types.TypeAndValue),
		Defs:         make(map[*ast.Ident]types.Object),
		Uses:         make(map[*ast.Ident]types.Object),
		Implicits:    make(map[ast.Node]types.Object),
		Instances:    make(map[*ast.Ident]types.Instance),
		Scopes:       make(map[ast.Node]*types.Scope),
		Selections:   make(map[*ast.SelectorExpr]*types.Selection),
		FileVersions: make(map[*ast.File]string),
	}
	if err := types.NewChecker(tc, fset, typPkg, info).Files([]*ast.File{f}); err != nil {
		t.Fatal(err)
	}

	// Build SSA (replicating ssautil.BuildPackage logic).
	prog := NewProgram(fset, SanityCheckFunctions)

	// Create SSA packages for all transitive imports.
	created := make(map[*types.Package]bool)
	var createAll func(pkgs []*types.Package)
	createAll = func(pkgs []*types.Package) {
		for _, p := range pkgs {
			if !created[p] {
				created[p] = true
				prog.CreatePackage(p, nil, nil, true)
				createAll(p.Imports())
			}
		}
	}
	createAll(typPkg.Imports())

	ssaPkg := prog.CreatePackage(typPkg, []*ast.File{f}, info, false)
	ssaPkg.Build()
	return ssaPkg
}

// TestSPMDCloneBlock_TranslateValue verifies that spmdTranslateValue returns
// the mapped value when present and the original value when not present.
func TestSPMDCloneBlock_TranslateValue(t *testing.T) {
	src := `package main

var result [8]int32

func main() {
	for i := range 8 {
		result[i] = int32(i * 2)
	}
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found")
	}

	// Collect the first two value-producing instructions to serve as map key/value.
	var v0, v1 Value
	for _, blk := range fn.Blocks {
		for _, instr := range blk.Instrs {
			val, ok := instr.(Value)
			if !ok {
				continue
			}
			if v0 == nil {
				v0 = val
			} else if v1 == nil {
				v1 = val
				break
			}
		}
		if v0 != nil && v1 != nil {
			break
		}
	}
	if v0 == nil || v1 == nil {
		t.Fatal("could not find two value-producing instructions")
	}

	valueMap := map[Value]Value{v0: v1}

	// v0 is in the map — must return v1.
	got := spmdTranslateValue(v0, valueMap)
	if got != v1 {
		t.Errorf("spmdTranslateValue(v0): got %v, want %v", got, v1)
	}

	// v1 is not in the map — must return v1 unchanged.
	got = spmdTranslateValue(v1, valueMap)
	if got != v1 {
		t.Errorf("spmdTranslateValue(v1, no-match): got %v, want %v", got, v1)
	}
}

// TestSPMDCloneBlock_Basic builds a simple SPMD program, finds the body block,
// clones it into a fresh block, and verifies:
//   - the clone contains exactly the expected number of non-phi, non-terminator
//     instructions
//   - all cloned instructions belong to the clone block, not the original
//   - valueMap entries point to instructions in the clone block
//   - the original body block is unmodified
func TestSPMDCloneBlock_Basic(t *testing.T) {
	src := `package main

var result [8]int32

func main() {
	for i := range 8 {
		result[i] = int32(i * 2)
	}
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found")
	}

	// Find the body block: for rangeint the comment contains "rangeint.body".
	var bodyBlock *BasicBlock
	for _, blk := range fn.Blocks {
		if strings.Contains(blk.Comment, "rangeint.body") {
			bodyBlock = blk
			break
		}
	}
	if bodyBlock == nil {
		var comments []string
		for _, blk := range fn.Blocks {
			comments = append(comments, blk.Comment)
		}
		t.Fatalf("rangeint.body block not found; available blocks: %v", comments)
	}

	// Count non-phi, non-terminator instructions in the source block.
	origCloneable := 0
	for _, instr := range bodyBlock.Instrs {
		switch instr.(type) {
		case *Phi, *Jump, *If, *DebugRef:
			// not cloned
		default:
			origCloneable++
		}
	}
	if origCloneable == 0 {
		t.Fatal("body block has no cloneable (non-phi, non-terminator) instructions")
	}

	// Snapshot original instruction count to verify it is unchanged after cloning.
	origTotal := len(bodyBlock.Instrs)

	// Create a fresh block for the clone and run spmdCloneBlock.
	cloneBlock := fn.newBasicBlock("clone")
	valueMap := make(map[Value]Value)
	spmdCloneBlock(fn, bodyBlock, cloneBlock, valueMap)

	// Original block must be unmodified.
	if len(bodyBlock.Instrs) != origTotal {
		t.Errorf("original block modified: had %d instrs, now %d", origTotal, len(bodyBlock.Instrs))
	}

	// Clone block must have exactly origCloneable instructions.
	if len(cloneBlock.Instrs) != origCloneable {
		t.Errorf("clone block has %d instrs, want %d", len(cloneBlock.Instrs), origCloneable)
	}

	// Every instruction in the clone block must belong to cloneBlock.
	for i, instr := range cloneBlock.Instrs {
		if instr.Block() != cloneBlock {
			t.Errorf("clone instr[%d] (%T) belongs to block %v, want clone block",
				i, instr, instr.Block())
		}
	}

	// Every value-producing clone must appear in valueMap and point to cloneBlock.
	for origVal, clonedVal := range valueMap {
		clonedInstr, ok := clonedVal.(Instruction)
		if !ok {
			t.Errorf("valueMap[%v] = %v is not an Instruction", origVal, clonedVal)
			continue
		}
		if clonedInstr.Block() != cloneBlock {
			t.Errorf("valueMap[%v].Block() = %v, want clone block", origVal, clonedInstr.Block())
		}
		// The original instruction must not point to the clone block.
		if origInstr, ok := origVal.(Instruction); ok {
			if origInstr.Block() == cloneBlock {
				t.Errorf("original instruction %v unexpectedly points to clone block", origVal)
			}
		}
	}

	// The valueMap must be non-empty (at least one value-producing instruction was cloned).
	if len(valueMap) == 0 {
		t.Error("valueMap is empty after cloning; expected at least one value-producing instruction")
	}
}

// TestSPMDCloneBlock_EmptyLikeBlock verifies that cloning a block containing
// only terminators/phis/debug-refs produces an empty clone and empty valueMap.
func TestSPMDCloneBlock_EmptyLikeBlock(t *testing.T) {
	src := `package main

func main() {
	for i := range 8 {
		_ = i
	}
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found")
	}

	// Find the done block — for rangeint it ends in a Return and has no
	// cloneable instructions.
	var doneBlock *BasicBlock
	for _, blk := range fn.Blocks {
		if strings.Contains(blk.Comment, "rangeint.done") {
			doneBlock = blk
			break
		}
	}
	if doneBlock == nil {
		t.Skip("rangeint.done block not found; skipping empty-block test")
	}

	cloneBlock := fn.newBasicBlock("clone-empty")
	valueMap := make(map[Value]Value)
	spmdCloneBlock(fn, doneBlock, cloneBlock, valueMap)

	// Done block has only a Return terminator (skipped) — clone must be empty.
	if len(cloneBlock.Instrs) != 0 {
		t.Errorf("expected 0 instrs in clone of done block, got %d", len(cloneBlock.Instrs))
	}
	if len(valueMap) != 0 {
		t.Errorf("expected empty valueMap for done block clone, got %d entries", len(valueMap))
	}
}

// TestSPMDCloneBlock_OperandTranslation verifies that operands of cloned
// instructions are translated through the valueMap: no cloned instruction
// should reference an original value that was itself cloned.
func TestSPMDCloneBlock_OperandTranslation(t *testing.T) {
	src := `package main

var result [8]int32

func main() {
	for i := range 8 {
		result[i] = int32(i * 2)
	}
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found")
	}

	// Find the body block.
	var bodyBlock *BasicBlock
	for _, blk := range fn.Blocks {
		if strings.Contains(blk.Comment, "rangeint.body") {
			bodyBlock = blk
			break
		}
	}
	if bodyBlock == nil {
		t.Fatal("rangeint.body block not found")
	}

	cloneBlock := fn.newBasicBlock("clone-ops")
	valueMap := make(map[Value]Value)
	spmdCloneBlock(fn, bodyBlock, cloneBlock, valueMap)

	// Build a set of original values that were cloned (i.e. keys of valueMap).
	origValues := make(map[Value]bool, len(valueMap))
	for orig := range valueMap {
		origValues[orig] = true
	}

	// For each cloned instruction, none of its operands should be an original
	// value that was itself cloned (it must have been translated to the clone).
	for _, instr := range cloneBlock.Instrs {
		ops := instr.Operands(nil)
		for _, opPtr := range ops {
			if opPtr == nil || *opPtr == nil {
				continue
			}
			op := *opPtr
			if origValues[op] {
				t.Errorf("cloned instr %T has operand %v that is the un-translated original",
					instr, op)
			}
		}
	}
}
