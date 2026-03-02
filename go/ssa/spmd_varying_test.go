package ssa_test

import (
	"bytes"
	"go/token"
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

	found := false
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok {
			if ifInstr.IsVarying {
				found = true
				break
			}
		}
	}
	if !found {
		t.Error("expected at least one If with IsVarying=true")
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

	varyingCount := 0
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && ifInstr.IsVarying {
			varyingCount++
		}
	}
	if varyingCount < 2 {
		t.Errorf("expected at least 2 varying If instructions for && condition, got %d", varyingCount)
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

	varyingCount := 0
	for _, block := range mainFn.Blocks {
		if len(block.Instrs) == 0 {
			continue
		}
		if ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If); ok && ifInstr.IsVarying {
			varyingCount++
		}
	}
	if varyingCount < 2 {
		t.Errorf("expected at least 2 varying If instructions for || condition, got %d", varyingCount)
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
	if !strings.Contains(output, "if varying") {
		t.Error("expected 'if varying' in SSA output for varying condition")
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
	// All chain blocks should share the same false successor (ElseBlock).
	for i, blk := range chain.Blocks {
		if blk.Succs[1] != chain.ElseBlock {
			t.Errorf("block %d (index %d): false successor is block %d, want %d",
				i, blk.Index, blk.Succs[1].Index, chain.ElseBlock.Index)
		}
	}
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
	// All chain blocks should share the same true successor (ThenBlock).
	for i, blk := range chain.Blocks {
		if blk.Succs[0] != chain.ThenBlock {
			t.Errorf("block %d: true successor is block %d, want %d",
				i, blk.Succs[0].Index, chain.ThenBlock.Index)
		}
	}
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
