package ssa_test

import (
	"bytes"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
)

func TestVaryingIf_Simple(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 5 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	// predicateSPMD linearizes varying Ifs into Jump + mask computations.
	// For an if-without-else with no Phi at the merge (only a discard `_ = i`),
	// no SPMDSelect is generated (there is no value to select). However, the
	// varying If must be gone (replaced by Jump) and mask instructions present.
	foundVaryingIf := false
	foundMaskConvert := false
	for _, block := range mainFn.Blocks {
		for _, instr := range block.Instrs {
			if ifInstr, ok := instr.(*ssa.If); ok && ifInstr.IsVarying {
				foundVaryingIf = true
			}
			// Mask computations: Convert(cond → Varying[mask]) is inserted by predicateSPMD.
			if conv, ok := instr.(*ssa.Convert); ok {
				if _, isMask := conv.Type().(*types.SPMDType); isMask {
					foundMaskConvert = true
				}
			}
		}
	}
	if foundVaryingIf {
		t.Error("unexpected IsVarying If: predicateSPMD should have linearized it to a Jump")
	}
	if !foundMaskConvert {
		t.Error("expected a Convert-to-Varying[mask] instruction inserted by predicateSPMD")
	}
}

func TestVaryingIf_UniformCondition(t *testing.T) {
	src := `package main

func main() {
	x := 3
	if x > 5 {
		_ = x
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if ifInstr.IsVarying {
				t.Errorf("block %d: If should not be varying for uniform condition", block.Index)
			}
		}
	}
}

func TestVaryingSwitch(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		switch i % 3 {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		default:
			_ = i + 2
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDSwitchChains) != 1 {
		t.Fatalf("expected 1 SPMDSwitchChain, got %d", len(mainFn.SPMDSwitchChains))
	}

	chain := mainFn.SPMDSwitchChains[0]
	if chain.TagValue == nil {
		t.Error("TagValue is nil")
	}
	if len(chain.Cases) < 2 {
		t.Errorf("expected at least 2 cases, got %d", len(chain.Cases))
	}
	if chain.DoneBlock == nil {
		t.Error("DoneBlock is nil")
	}

	// All cases should be marked varying.
	for i, ifInstr := range chain.Cases {
		if !ifInstr.IsVarying {
			t.Errorf("case %d: If instruction not marked varying", i)
		}
	}
}

func TestUniformSwitch(t *testing.T) {
	src := `package main

func main() {
	x := 3
	switch x {
	case 1:
		_ = x
	case 2:
		_ = x + 1
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDSwitchChains) != 0 {
		t.Errorf("expected 0 SPMDSwitchChains for uniform switch, got %d", len(mainFn.SPMDSwitchChains))
	}
}

func TestVaryingIf_CompoundAnd(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	// predicateBooleanChain runs as part of build and linearizes all chain Ifs.
	// After predication no varying If instructions should remain.
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && ifInstr.IsVarying {
			t.Errorf("block %d: varying If remains after boolean chain && predication", block.Index)
		}
	}

	// The SPMDBooleanChain metadata must still be present and record the && pattern.
	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}
	if mainFn.SPMDBooleanChains[0].Op != token.LAND {
		t.Errorf("expected LAND chain op, got %v", mainFn.SPMDBooleanChains[0].Op)
	}
}

func TestVaryingIf_CompoundOr(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i < 2 || i > 10 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	// predicateBooleanChain runs as part of build and linearizes all chain Ifs.
	// After predication no varying If instructions should remain.
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && ifInstr.IsVarying {
			t.Errorf("block %d: varying If remains after boolean chain || predication", block.Index)
		}
	}

	// The SPMDBooleanChain metadata must still be present and record the || pattern.
	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}
	if mainFn.SPMDBooleanChains[0].Op != token.LOR {
		t.Errorf("expected LOR chain op, got %v", mainFn.SPMDBooleanChains[0].Op)
	}
}

func TestVaryingSwitch_ConvertStripsType(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		switch byte(i % 3) {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDSwitchChains) != 1 {
		t.Fatalf("expected 1 SPMDSwitchChain for byte(varyingExpr) switch, got %d",
			len(mainFn.SPMDSwitchChains))
	}
}

func TestVaryingIf_String(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 5 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()

	// predicateSPMD linearizes "if varying" into a Jump + mask computations.
	// The "if varying" annotation should be gone after transformation.
	// For an if-without-else with no live Phi at the merge, no spmd_select is
	// generated; instead, a Varying[mask] Convert instruction is emitted.
	if strings.Contains(output, "if varying") {
		t.Error("unexpected 'if varying' in SSA output: predicateSPMD should have linearized it")
	}
	if !strings.Contains(output, "lanes.Varying[mask]") {
		t.Error("expected 'lanes.Varying[mask]' mask computation in SSA output after predicateSPMD")
	}
}

func TestVaryingSwitch_DefaultBlockResolution(t *testing.T) {
	// The default case body block may be fused into the last switch.next
	// block by optimizeBlocks (fuseBlocks). Verify that SPMDSwitchChain
	// block pointers are resolved to surviving blocks after optimization.
	src := `package main

func main() {
	for i := range 16 {
		switch i % 4 {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		case 2:
			_ = i + 2
		default:
			_ = i + 3
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDSwitchChains) != 1 {
		t.Fatalf("expected 1 SPMDSwitchChain, got %d", len(mainFn.SPMDSwitchChains))
	}

	chain := mainFn.SPMDSwitchChains[0]

	// DefaultBlock must point to a valid block in the function.
	if chain.DefaultBlock == nil {
		t.Fatal("DefaultBlock is nil, expected non-nil for switch with default case")
	}
	found := false
	for _, blk := range mainFn.Blocks {
		if blk == chain.DefaultBlock {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("DefaultBlock (index %d, comment %q) not found in function blocks",
			chain.DefaultBlock.Index, chain.DefaultBlock.Comment)
	}

	// DoneBlock must also be valid.
	found = false
	for _, blk := range mainFn.Blocks {
		if blk == chain.DoneBlock {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("DoneBlock (index %d, comment %q) not found in function blocks",
			chain.DoneBlock.Index, chain.DoneBlock.Comment)
	}

	// DefaultBlock.Index must be within range.
	if chain.DefaultBlock.Index >= len(mainFn.Blocks) {
		t.Errorf("DefaultBlock.Index=%d out of range for %d blocks",
			chain.DefaultBlock.Index, len(mainFn.Blocks))
	}
}

func TestVaryingSwitchChain_String(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		switch i % 3 {
		case 0:
			_ = i
		case 1:
			_ = i + 1
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "SPMDSwitchChain") {
		t.Error("expected 'SPMDSwitchChain' in SSA output for varying switch")
	}
}

func TestBooleanChain_And(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}

	chain := mainFn.SPMDBooleanChains[0]
	if chain.Op != token.LAND {
		t.Errorf("expected LAND, got %v", chain.Op)
	}
	if len(chain.Blocks) != 2 {
		t.Errorf("expected 2 blocks in chain, got %d", len(chain.Blocks))
	}
	if chain.ThenBlock == nil {
		t.Error("ThenBlock is nil")
	}
	if chain.ElseBlock == nil {
		t.Error("ElseBlock is nil")
	}
	if !chain.IsVarying {
		t.Error("expected IsVarying=true")
	}
	// Note: Succs structure is checked before predicateBooleanChain runs.
	// After predication, blocks have Jump (1 successor) instead of If (2 successors),
	// so Succs[1] is no longer valid. The pre-predication invariant is validated
	// by the metadata (ThenBlock/ElseBlock) captured in SPMDBooleanChain.
}

func TestBooleanChain_Or(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i < 2 || i > 10 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}

	chain := mainFn.SPMDBooleanChains[0]
	if chain.Op != token.LOR {
		t.Errorf("expected LOR, got %v", chain.Op)
	}
	if len(chain.Blocks) != 2 {
		t.Errorf("expected 2 blocks in chain, got %d", len(chain.Blocks))
	}
	if !chain.IsVarying {
		t.Error("expected IsVarying=true")
	}
	// Note: Succs structure is checked before predicateBooleanChain runs.
	// After predication, blocks have Jump (1 successor) instead of If (2 successors),
	// so Succs[0] is the jump target, not the pre-predication true successor.
	// The pre-predication invariant is validated by ThenBlock in SPMDBooleanChain.
}

func TestBooleanChain_TripleAnd(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 && i != 5 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}

	chain := mainFn.SPMDBooleanChains[0]
	if chain.Op != token.LAND {
		t.Errorf("expected LAND, got %v", chain.Op)
	}
	if len(chain.Blocks) != 3 {
		t.Errorf("expected 3 blocks in chain, got %d", len(chain.Blocks))
	}
}

func TestBooleanChain_TripleOr(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i < 2 || i > 10 || i == 5 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}

	chain := mainFn.SPMDBooleanChains[0]
	if chain.Op != token.LOR {
		t.Errorf("expected LOR, got %v", chain.Op)
	}
	if len(chain.Blocks) != 3 {
		t.Errorf("expected 3 blocks in chain, got %d", len(chain.Blocks))
	}
}

func TestBooleanChain_MixedAndOr(t *testing.T) {
	// (a && b) || c produces inner LAND chain only.
	// The outer LOR is not a flat chain (LHS is compound).
	src := `package main

func main() {
	for i := range 16 {
		if (i > 2 && i < 10) || i == 0 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	// Should have exactly 1 chain: the inner LAND.
	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain for (a&&b)||c, got %d", len(mainFn.SPMDBooleanChains))
	}

	chain := mainFn.SPMDBooleanChains[0]
	if chain.Op != token.LAND {
		t.Errorf("expected inner chain to be LAND, got %v", chain.Op)
	}
	if len(chain.Blocks) != 2 {
		t.Errorf("expected 2 blocks in inner LAND chain, got %d", len(chain.Blocks))
	}
}

func TestBooleanChain_UniformCondition(t *testing.T) {
	// Uniform conditions should still create a chain but with IsVarying=false.
	src := `package main

func main() {
	x := 3
	y := 7
	if x > 2 && y < 10 {
		_ = x
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 SPMDBooleanChain, got %d", len(mainFn.SPMDBooleanChains))
	}

	chain := mainFn.SPMDBooleanChains[0]
	if chain.IsVarying {
		t.Error("expected IsVarying=false for uniform conditions")
	}
}

func TestBooleanChain_NotExclusion(t *testing.T) {
	// !a && b should NOT form a chain (NOT inverts successor pattern).
	src := `package main

func main() {
	for i := range 16 {
		if !(i > 5) && i < 10 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	// No chain because NOT breaks the shared-target invariant.
	if len(mainFn.SPMDBooleanChains) != 0 {
		t.Errorf("expected 0 SPMDBooleanChains for !a && b, got %d", len(mainFn.SPMDBooleanChains))
	}
}

func TestBooleanChain_String(t *testing.T) {
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	var buf bytes.Buffer
	mainFn.WriteTo(&buf)
	output := buf.String()
	if !strings.Contains(output, "SPMDBooleanChain") {
		t.Error("expected 'SPMDBooleanChain' in SSA output for && chain")
	}
}

func TestBooleanChain_BlockResolution(t *testing.T) {
	// Verify that chain block pointers survive optimizeBlocks.
	src := `package main

func main() {
	for i := range 16 {
		if i > 2 && i < 10 && i != 5 {
			_ = i
		}
	}
}
`
	pkg := buildSSAWithSPMD(t, src)
	mainFn := pkg.Func("main")

	if len(mainFn.SPMDBooleanChains) != 1 {
		t.Fatalf("expected 1 chain, got %d", len(mainFn.SPMDBooleanChains))
	}
	chain := mainFn.SPMDBooleanChains[0]

	// ThenBlock and ElseBlock must be valid blocks in the function.
	for _, target := range []*ssa.BasicBlock{chain.ThenBlock, chain.ElseBlock} {
		found := false
		for _, blk := range mainFn.Blocks {
			if blk == target {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("block %d (comment %q) not found in function blocks",
				target.Index, target.Comment)
		}
	}

	// All chain blocks must also be in the function.
	for i, blk := range chain.Blocks {
		found := false
		for _, fblk := range mainFn.Blocks {
			if fblk == blk {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("chain block %d (index %d) not found in function blocks", i, blk.Index)
		}
	}
}
