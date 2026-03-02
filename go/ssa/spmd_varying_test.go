package ssa_test

import (
	"bytes"
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
