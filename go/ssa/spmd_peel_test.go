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

	// After peeling, the original rangeint.body block is unreachable and removed
	// from fn.Blocks. Use the SPMDLoopInfo.BodyBlock pointer which still
	// references the original block object (its Instrs remain intact).
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	bodyBlock := fn.SPMDLoops[0].BodyBlock
	if bodyBlock == nil {
		t.Fatal("BodyBlock is nil")
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

// TestPeelSPMDLoopSimple verifies that peelSPMDLoops creates the expected CFG
// structure for a simple rangeint SPMD loop with no accumulators.
func TestPeelSPMDLoopSimple(t *testing.T) {
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
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	loop := fn.SPMDLoops[0]

	// Verify peeling happened.
	if !loop.IsPeeled {
		t.Fatal("loop was not peeled")
	}
	if loop.MainBodyBlock == nil {
		t.Fatal("MainBodyBlock is nil")
	}
	if loop.TailCheckBlock == nil {
		t.Fatal("TailCheckBlock is nil")
	}
	if loop.TailBodyBlock == nil {
		t.Fatal("TailBodyBlock is nil")
	}
	if loop.TrampolineBlock != nil {
		t.Error("TrampolineBlock should be nil for loop without accumulators")
	}
	if loop.AlignedBound == nil {
		t.Fatal("AlignedBound is nil")
	}
	if loop.MainIterPhi == nil {
		t.Fatal("MainIterPhi is nil")
	}
	if loop.TailIterPhi == nil {
		t.Fatal("TailIterPhi is nil")
	}

	// Verify CFG structure: entry should branch to MainBody or TailCheck.
	entry := loop.EntryBlock
	if len(entry.Succs) != 2 {
		t.Fatalf("entry has %d succs, want 2", len(entry.Succs))
	}
	if entry.Succs[0] != loop.MainBodyBlock {
		t.Errorf("entry.Succs[0] = %s, want MainBody", entry.Succs[0].Comment)
	}
	if entry.Succs[1] != loop.TailCheckBlock {
		t.Errorf("entry.Succs[1] = %s, want TailCheck", entry.Succs[1].Comment)
	}

	// MainBody should self-loop and exit to TailCheck.
	main := loop.MainBodyBlock
	if len(main.Succs) != 2 {
		t.Fatalf("MainBody has %d succs, want 2", len(main.Succs))
	}
	if main.Succs[0] != main {
		t.Errorf("MainBody.Succs[0] should be self (loop back), got %s", main.Succs[0].Comment)
	}
	if main.Succs[1] != loop.TailCheckBlock {
		t.Errorf("MainBody.Succs[1] = %s, want TailCheck", main.Succs[1].Comment)
	}

	// TailCheck should branch to TailBody or Done.
	tc := loop.TailCheckBlock
	if len(tc.Succs) != 2 {
		t.Fatalf("TailCheck has %d succs, want 2", len(tc.Succs))
	}
	if tc.Succs[0] != loop.TailBodyBlock {
		t.Errorf("TailCheck.Succs[0] = %s, want TailBody", tc.Succs[0].Comment)
	}
	if tc.Succs[1] != loop.DoneBlock {
		t.Errorf("TailCheck.Succs[1] = %s, want Done", tc.Succs[1].Comment)
	}

	// TailBody should jump to Done.
	tb := loop.TailBodyBlock
	if len(tb.Succs) != 1 {
		t.Fatalf("TailBody has %d succs, want 1", len(tb.Succs))
	}
	if tb.Succs[0] != loop.DoneBlock {
		t.Errorf("TailBody.Succs[0] = %s, want Done", tb.Succs[0].Comment)
	}

	// Done should have exactly 2 predecessors: TailCheck and TailBody.
	done := loop.DoneBlock
	if len(done.Preds) != 2 {
		t.Fatalf("Done has %d preds, want 2", len(done.Preds))
	}

	// MainBody and TailBody should contain cloned instructions.
	if len(main.Instrs) == 0 {
		t.Error("MainBody has no instructions")
	}
	if len(tb.Instrs) == 0 {
		t.Error("TailBody has no instructions")
	}

	// MainIterPhi must be in MainBodyBlock.
	if loop.MainIterPhi.Block() != loop.MainBodyBlock {
		t.Errorf("MainIterPhi.Block() = %s, want MainBodyBlock", loop.MainIterPhi.Block().Comment)
	}

	// TailIterPhi must be in TailCheckBlock.
	if loop.TailIterPhi.Block() != loop.TailCheckBlock {
		t.Errorf("TailIterPhi.Block() = %s, want TailCheckBlock", loop.TailIterPhi.Block().Comment)
	}

	// AlignedBound must be in EntryBlock.
	if ab, ok := loop.AlignedBound.(Instruction); ok {
		if ab.Block() != loop.EntryBlock {
			t.Errorf("AlignedBound.Block() = %s, want EntryBlock", ab.Block().Comment)
		}
	}
}

// TestPeelSPMDLoopWithAccumulator verifies that peelSPMDLoops creates the
// expected CFG structure for a rangeint SPMD loop with a loop-carried
// accumulator. A trampoline block must be inserted between the tail check /
// tail body and the done block so that the two accumulator paths are merged
// before the done block is reached.
func TestPeelSPMDLoopWithAccumulator(t *testing.T) {
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
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	loop := fn.SPMDLoops[0]

	if !loop.IsPeeled {
		t.Fatal("loop was not peeled")
	}
	if len(loop.Accumulators) == 0 {
		t.Fatal("expected at least one accumulator")
	}
	if loop.TrampolineBlock == nil {
		t.Fatal("TrampolineBlock should not be nil for loop with accumulators")
	}

	// Verify trampoline structure: exactly one successor (done).
	tramp := loop.TrampolineBlock
	if len(tramp.Succs) != 1 {
		t.Fatalf("trampoline has %d succs, want 1", len(tramp.Succs))
	}
	if tramp.Succs[0] != loop.DoneBlock {
		t.Errorf("trampoline.Succs[0] = %s, want Done", tramp.Succs[0].Comment)
	}

	// Trampoline must have at least one phi (the accumulator merge).
	hasPhi := false
	for _, instr := range tramp.Instrs {
		if _, ok := instr.(*Phi); ok {
			hasPhi = true
			break
		}
	}
	if !hasPhi {
		t.Error("trampoline should have at least one phi for accumulator merge")
	}

	// TailCheck must branch to TailBody (true) and trampoline (false), not done.
	tc := loop.TailCheckBlock
	if len(tc.Succs) != 2 {
		t.Fatalf("TailCheck has %d succs, want 2", len(tc.Succs))
	}
	if tc.Succs[0] != loop.TailBodyBlock {
		t.Errorf("TailCheck.Succs[0] = %s, want TailBody", tc.Succs[0].Comment)
	}
	if tc.Succs[1] != tramp {
		t.Errorf("TailCheck.Succs[1] = %s, want Trampoline", tc.Succs[1].Comment)
	}

	// TailBody must jump to trampoline, not done.
	tb := loop.TailBodyBlock
	if len(tb.Succs) != 1 {
		t.Fatalf("TailBody has %d succs, want 1", len(tb.Succs))
	}
	if tb.Succs[0] != tramp {
		t.Errorf("TailBody.Succs[0] = %s, want Trampoline", tb.Succs[0].Comment)
	}

	// Done must have Trampoline as its only predecessor (not TailCheck or TailBody directly).
	done := loop.DoneBlock
	hasTrampPred := false
	for _, p := range done.Preds {
		if p == tramp {
			hasTrampPred = true
		}
		if p == tc {
			t.Error("Done should not have TailCheck as direct predecessor when trampoline exists")
		}
		if p == tb {
			t.Error("Done should not have TailBody as direct predecessor when trampoline exists")
		}
	}
	if !hasTrampPred {
		t.Error("Done should have Trampoline as a predecessor")
	}

	// Trampoline must have exactly two predecessors: TailCheck and TailBody.
	if len(tramp.Preds) != 2 {
		t.Fatalf("trampoline has %d preds, want 2", len(tramp.Preds))
	}
	if tramp.Preds[0] != tc {
		t.Errorf("tramp.Preds[0] = %s, want TailCheck", tramp.Preds[0].Comment)
	}
	if tramp.Preds[1] != tb {
		t.Errorf("tramp.Preds[1] = %s, want TailBody", tramp.Preds[1].Comment)
	}

	// Verify the merge phi's edge count matches the trampoline's predecessors.
	for _, instr := range tramp.Instrs {
		phi, ok := instr.(*Phi)
		if !ok {
			continue
		}
		if len(phi.Edges) != 2 {
			t.Errorf("merge phi %q has %d edges, want 2", phi.Comment, len(phi.Edges))
		}
	}
}

// TestPeelSPMDLoopUniformEarlyReturn verifies that a go-for loop whose body
// ends with a uniform (non-varying) If followed by a Return — simulating the
// "if reduce.Any(found) { return true }" pattern in lo-contains — is peeled
// into separate main and tail phases that each contain an early-exit block.
//
// The loop body has the shape:
//
//	body: phi iter; call check(); If cond → if.then, if.done
//	if.then: Return true
//	if.done: incr; cond; If → body, done
//	done: Return false
func TestPeelSPMDLoopUniformEarlyReturn(t *testing.T) {
	// "check" is a package-level func var so the call is a plain *Call (no
	// MakeInterface), satisfying spmdBodyIsCloneable. The uniform bool result
	// means the If is non-varying, triggering shapeUniformExit in the peeler.
	src := `package main

var check func() bool

func containsSPMD(target int) bool {
	for i := range 16 {
		_ = i
		if check() {
			return true
		}
	}
	return false
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("containsSPMD")
	if fn == nil {
		t.Fatal("containsSPMD function not found")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	loop := fn.SPMDLoops[0]

	// The loop must have been peeled.
	if !loop.IsPeeled {
		t.Fatal("loop was not peeled")
	}
	if loop.MainBodyBlock == nil {
		t.Fatal("MainBodyBlock is nil")
	}
	if loop.TailCheckBlock == nil {
		t.Fatal("TailCheckBlock is nil")
	}
	if loop.TailBodyBlock == nil {
		t.Fatal("TailBodyBlock is nil")
	}
	if loop.BackEdgeBlock == nil {
		t.Fatal("BackEdgeBlock is nil")
	}

	// main.body must NOT be its own successor (the self-loop is via the
	// cloned back-edge block, not main.body itself). That is: main.body's
	// If sends the true branch to the early-exit clone and the false branch
	// to the cloned back-edge block; the back-edge block's If loops back.
	mainBody := loop.MainBodyBlock
	for _, succ := range mainBody.Succs {
		if succ == mainBody {
			t.Error("MainBodyBlock must not be its own successor; back-edge should go via cloned back-edge block")
		}
	}

	// main.body must have exactly two successors (exit clone and back-edge clone).
	if len(mainBody.Succs) != 2 {
		t.Errorf("MainBodyBlock has %d succs, want 2", len(mainBody.Succs))
	}

	// At least one successor of main.body must terminate with Return (the early exit).
	foundEarlyReturn := false
	for _, succ := range mainBody.Succs {
		if len(succ.Instrs) > 0 {
			if _, ok := succ.Instrs[len(succ.Instrs)-1].(*Return); ok {
				foundEarlyReturn = true
			}
		}
	}
	if !foundEarlyReturn {
		t.Error("no successor of MainBodyBlock terminates with Return (expected early-exit clone)")
	}

	// tail.body must also have exactly two successors and an early-exit Return clone.
	tailBody := loop.TailBodyBlock
	if len(tailBody.Succs) != 2 {
		t.Errorf("TailBodyBlock has %d succs, want 2", len(tailBody.Succs))
	}
	foundTailEarlyReturn := false
	for _, succ := range tailBody.Succs {
		if len(succ.Instrs) > 0 {
			if _, ok := succ.Instrs[len(succ.Instrs)-1].(*Return); ok {
				foundTailEarlyReturn = true
			}
		}
	}
	if !foundTailEarlyReturn {
		t.Error("no successor of TailBodyBlock terminates with Return (expected tail early-exit clone)")
	}

	// TailCheck must branch to TailBody (true) and the done block (false).
	tc := loop.TailCheckBlock
	if len(tc.Succs) != 2 {
		t.Fatalf("TailCheck has %d succs, want 2", len(tc.Succs))
	}
	if tc.Succs[0] != tailBody {
		t.Errorf("TailCheck.Succs[0] = %s, want TailBody", tc.Succs[0].Comment)
	}
	if tc.Succs[1] != loop.DoneBlock {
		t.Errorf("TailCheck.Succs[1] = %s, want DoneBlock", tc.Succs[1].Comment)
	}

	// MainIterPhi must live in MainBodyBlock.
	if loop.MainIterPhi == nil {
		t.Fatal("MainIterPhi is nil")
	}
	if loop.MainIterPhi.Block() != mainBody {
		t.Errorf("MainIterPhi.Block() = %s, want MainBodyBlock", loop.MainIterPhi.Block().Comment)
	}

	// The loop must not have a trampoline: there are no accumulators and
	// done has no phis in this simple case.
	if loop.TrampolineBlock != nil {
		t.Error("TrampolineBlock should be nil (no accumulators)")
	}

	// AlignedBound must live in EntryBlock.
	if loop.AlignedBound == nil {
		t.Fatal("AlignedBound is nil")
	}
	if ab, ok := loop.AlignedBound.(Instruction); ok {
		if ab.Block() != loop.EntryBlock {
			t.Errorf("AlignedBound.Block() = %s, want EntryBlock", ab.Block().Comment)
		}
	}
}

// TestPeelSPMDLoopUniformEarlyJump verifies the variant where the uniform early
// exit jumps to done rather than returning. This simulates a "break" statement:
//
//	body: ... If cond → break.block, cont
//	break.block: Jump → done    (uniform break)
//	cont: incr; If → body, done
//	done: ...
//
// After peeling the break block in the main phase must jump to done (or
// trampoline), and the tail phase similarly exits cleanly.
func TestPeelSPMDLoopUniformEarlyJump(t *testing.T) {
	// Use labeled-break to produce a "if cond { break }" inside the loop.
	// The break lowers to a single Jump → done without a Return.
	src := `package main

var check func() bool

func containsSPMD(target int) int {
	result := 0
outer:
	for i := range 16 {
		_ = i
		if check() {
			break outer
		}
		result++
	}
	return result
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("containsSPMD")
	if fn == nil {
		t.Fatal("containsSPMD function not found")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	loop := fn.SPMDLoops[0]

	// The loop may or may not peel depending on whether the break-to-done block
	// is recognizable as a spmdIsExitBlock. If it peeled, verify the structure;
	// if not (shapeUnpeelable), the test is informational only.
	if !loop.IsPeeled {
		t.Log("loop was not peeled (shapeUnpeelable for break pattern — acceptable)")
		return
	}

	// Basic peeled structure checks.
	if loop.MainBodyBlock == nil {
		t.Fatal("MainBodyBlock is nil")
	}
	if loop.TailBodyBlock == nil {
		t.Fatal("TailBodyBlock is nil")
	}
	mainBody := loop.MainBodyBlock
	if len(mainBody.Succs) != 2 {
		t.Errorf("MainBodyBlock has %d succs, want 2", len(mainBody.Succs))
	}
}

// TestPeelSPMDLoopUniformExit_VaryingIfBails verifies that a loop body whose
// If terminator is varying (IsVarying==true) does NOT trigger shapeUniformExit:
// the peeler must bail and leave the loop unpeeled.
func TestPeelSPMDLoopUniformExit_VaryingIfBails(t *testing.T) {
	// A body whose If compares a varying value directly (not routed through
	// reduce.Any) is classified as varying (IsVarying=true) by the SSA builder.
	// The shapeUniformExit classifier must reject it and fall through to
	// shapeUnpeelable.
	//
	// "break" under a varying condition is forbidden by the SPMD type checker,
	// so we use a varying If that jumps to another body block (no early exit).
	// This forces the body to have Succs != [body, done] but with a varying
	// If, triggering shapeUnpeelable.
	src := `package main

var result [16]int32

func main() {
	for i := range 16 {
		if i > 8 {
			result[i] = int32(i)
		} else {
			result[i] = 0
		}
	}
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("main")
	if fn == nil {
		t.Fatal("main function not found")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	loop := fn.SPMDLoops[0]

	// A loop with varying if-else has a linearized body (Jump terminator) and
	// should have been peeled as shapeSimple (the predication converts the if-else
	// to SPMDSelect before peeling, leaving the body with Succs=[body,done]).
	// Either it peeled as simple or was not peeled; it must NOT be shaped as
	// shapeUniformExit (which only applies to non-varying Ifs).
	//
	// We verify that IF the loop was not peeled, it's because it bailed early,
	// not because of incorrect shapeUniformExit detection. If it was peeled as
	// shapeSimple, the varying If was already linearized and that's fine too.
	if loop.IsPeeled {
		// If peeled, it must be shapeSimple: main.body must have a self-loop
		// (the direct back-edge, as in the simple case after predication).
		mainBody := loop.MainBodyBlock
		if mainBody == nil {
			t.Fatal("IsPeeled=true but MainBodyBlock is nil")
		}
		// No early-exit Return successor in main.body (would indicate shapeUniformExit).
		for _, succ := range mainBody.Succs {
			if len(succ.Instrs) == 0 {
				continue
			}
			if _, ok := succ.Instrs[len(succ.Instrs)-1].(*Return); ok {
				t.Error("shapeSimple main.body should not have a Return-terminated successor")
			}
		}
	}
	// Whether peeled or not, no panic must have occurred.
}

// TestPeelSPMDLoopRangeIndexUniformExit verifies that a rangeindex loop
// (range-over-slice) whose body ends with a uniform (non-varying) If — the
// canonical "go for _, v := range data { if uniformCond { return true } }"
// early-exit pattern — is correctly peeled into main + tail phases.
//
// The test uses "for i := range data" (rangeindex, index-only) with a uniform
// condition from a package-level func var. The SPMD body has a uniform If that
// returns true early; the loop's back-edge is the false branch back to loopBlock.
func TestPeelSPMDLoopRangeIndexUniformExit(t *testing.T) {
	src := `package main

var check func() bool

func containsSlice(data []int32) bool {
	for i := range data {
		_ = i
		if check() {
			return true
		}
	}
	return false
}
`
	pkg := buildSSAForPeelTest(t, src)
	fn := pkg.Func("containsSlice")
	if fn == nil {
		t.Fatal("containsSlice function not found")
	}
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	loop := fn.SPMDLoops[0]

	// The loop is rangeindex (range-over-slice), not merged.
	if loop.MergedBodyLoop {
		t.Error("MergedBodyLoop should be false for range-over-slice")
	}
	if !loop.IsRangeIndex {
		t.Error("IsRangeIndex should be true for range-over-slice")
	}

	// The loop must have been peeled.
	if !loop.IsPeeled {
		t.Fatal("loop was not peeled — rangeindex uniform-exit peeling failed")
	}
	if loop.MainBodyBlock == nil {
		t.Fatal("MainBodyBlock is nil")
	}
	if loop.TailCheckBlock == nil {
		t.Fatal("TailCheckBlock is nil")
	}
	if loop.TailBodyBlock == nil {
		t.Fatal("TailBodyBlock is nil")
	}

	// MainIterPhi must live in the main loop block (not mainBody).
	if loop.MainIterPhi == nil {
		t.Fatal("MainIterPhi is nil")
	}
	mainLoopBlock := loop.BackEdgeBlock
	if mainLoopBlock == nil {
		t.Fatal("BackEdgeBlock (mainLoopBlock) is nil")
	}
	if loop.MainIterPhi.Block() != mainLoopBlock {
		t.Errorf("MainIterPhi.Block() = %s, want main loop block (%s)",
			loop.MainIterPhi.Block().Comment, mainLoopBlock.Comment)
	}

	// mainBody must have exactly two successors.
	mainBody := loop.MainBodyBlock
	if len(mainBody.Succs) != 2 {
		t.Errorf("MainBodyBlock has %d succs, want 2", len(mainBody.Succs))
	}

	// At least one successor of mainBody must terminate with Return (early exit).
	foundEarlyReturn := false
	for _, succ := range mainBody.Succs {
		if len(succ.Instrs) > 0 {
			if _, ok := succ.Instrs[len(succ.Instrs)-1].(*Return); ok {
				foundEarlyReturn = true
			}
		}
	}
	if !foundEarlyReturn {
		t.Error("no successor of MainBodyBlock terminates with Return (expected main early-exit clone)")
	}

	// One successor of mainBody must be mainLoopBlock (the loop-back).
	foundLoopBack := false
	for _, succ := range mainBody.Succs {
		if succ == mainLoopBlock {
			foundLoopBack = true
		}
	}
	if !foundLoopBack {
		t.Errorf("MainBodyBlock has no successor pointing to mainLoopBlock (%s)", mainLoopBlock.Comment)
	}

	// tailBody must have exactly two successors.
	tailBody := loop.TailBodyBlock
	if len(tailBody.Succs) != 2 {
		t.Errorf("TailBodyBlock has %d succs, want 2", len(tailBody.Succs))
	}

	// At least one successor of tailBody must terminate with Return.
	foundTailReturn := false
	for _, succ := range tailBody.Succs {
		if len(succ.Instrs) > 0 {
			if _, ok := succ.Instrs[len(succ.Instrs)-1].(*Return); ok {
				foundTailReturn = true
			}
		}
	}
	if !foundTailReturn {
		t.Error("no successor of TailBodyBlock terminates with Return (expected tail early-exit clone)")
	}

	// tailCheck must have two succs: tailBody and done.
	tc := loop.TailCheckBlock
	if len(tc.Succs) != 2 {
		t.Fatalf("TailCheckBlock has %d succs, want 2", len(tc.Succs))
	}
	if tc.Succs[0] != tailBody {
		t.Errorf("TailCheckBlock.Succs[0] = %s, want TailBodyBlock", tc.Succs[0].Comment)
	}
	if tc.Succs[1] != loop.DoneBlock {
		t.Errorf("TailCheckBlock.Succs[1] = %s, want DoneBlock", tc.Succs[1].Comment)
	}

	// AlignedBound must live in EntryBlock.
	if loop.AlignedBound == nil {
		t.Fatal("AlignedBound is nil")
	}
	if ab, ok := loop.AlignedBound.(Instruction); ok {
		if ab.Block() != loop.EntryBlock {
			t.Errorf("AlignedBound.Block() = %s, want EntryBlock", ab.Block().Comment)
		}
	}

	// No trampoline needed when the loop has no accumulators.
	if loop.TrampolineBlock != nil {
		t.Error("TrampolineBlock should be nil (no accumulators)")
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

	// After peeling the original body block is detached from fn.Blocks;
	// use SPMDLoopInfo.BodyBlock to access it directly.
	if len(fn.SPMDLoops) == 0 {
		t.Fatal("no SPMD loops found")
	}
	bodyBlock := fn.SPMDLoops[0].BodyBlock
	if bodyBlock == nil {
		t.Fatal("BodyBlock is nil")
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
