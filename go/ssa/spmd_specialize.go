// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"go/token"
)

// spmdSpecializePackage runs SPMD function specialization for a single package
// p in isolation. It is safe to call from the per-package p.build() path
// because it only reads and writes p.Members (not other packages' state),
// which is owned exclusively by p's build goroutine.
//
// This is the path used by ssautil.BuildPackage (which calls p.Build()
// directly without a subsequent prog.Build()). The prog.Build() path calls
// spmdSpecializeFunctions instead, which handles cross-package specialization
// after all packages have finished building.
func spmdSpecializePackage(p *Package) error {
	const maxIter = 10
	for iter := 0; iter < maxIter; iter++ {
		// Gather requests from functions in this package only.
		requests := make(map[*Function]map[int]bool)
		for _, mem := range p.Members {
			if mem == nil {
				continue
			}
			fn, ok := mem.(*Function)
			if !ok || fn == nil || fn.Blocks == nil {
				continue
			}
			spmdGatherRequestsInFunc(fn, requests)
		}
		if len(requests) == 0 {
			return nil
		}
		anyChange := spmdMaterializeVariantsForPackage(p, requests)
		if !anyChange {
			return nil
		}
	}
	return fmt.Errorf("spmdSpecializePackage: did not reach fixed point after %d iterations", maxIter)
}

// spmdMaterializeVariantsForPackage is like spmdMaterializeVariants but
// rewrites call sites only within package p.
func spmdMaterializeVariantsForPackage(p *Package, requests map[*Function]map[int]bool) bool {
	anyChange := false
	variants := make(map[*Function]map[int]*Function)

	for orig, lcSet := range requests {
		if orig.Pkg == nil || orig.Blocks == nil {
			continue
		}
		vmap := make(map[int]*Function)

		if len(lcSet) == 1 {
			var lc int
			for k := range lcSet {
				lc = k
			}
			newName := fmt.Sprintf("%s.spmd%d", orig.name, lc)
			delete(orig.Pkg.Members, orig.name)
			orig.name = newName
			orig.object = nil
			orig.Synthetic = "SPMD specialization"
			orig.syntax = nil
			orig.Pkg.Members[newName] = orig
			for _, bb := range orig.Blocks {
				if bb.SPMDLaneCount == 0 {
					bb.SPMDLaneCount = lc
				}
			}
			vmap[lc] = orig
			anyChange = true
		} else {
			for lc := range lcSet {
				clone := spmdCloneFunction(orig, lc)
				if clone == nil {
					continue
				}
				orig.Pkg.Members[clone.name] = clone
				vmap[lc] = clone
				anyChange = true
			}
			if len(vmap) > 0 {
				delete(orig.Pkg.Members, orig.name)
			}
		}
		variants[orig] = vmap
	}

	// Rewrite call sites within this package.
	for _, mem := range p.Members {
		if mem == nil {
			continue
		}
		fn, ok := mem.(*Function)
		if !ok || fn == nil || fn.Blocks == nil {
			continue
		}
		spmdRewriteCallSites(fn, variants)
	}

	return anyChange
}

// spmdSpecializeFunctions clones SPMD functions (those with lanes.Varying[T]
// parameters) once per unique caller lane count, so that TinyGo's block-
// annotation-aware materialization always sees a consistent SPMDLaneCount on
// every block inside the variant.
//
// Single-lane-count shortcut: if a callee is only ever requested at one lane
// count, it is renamed in place (suffix ".spmd<N>") rather than cloned. This
// avoids the cloning machinery entirely for the common case and keeps the SSA
// delta minimal.
//
// Multi-lane-count case: one clone per unique lane count is created. Each
// clone gets its blocks annotated with the variant's lane count. Call sites
// are rewritten to dispatch to the correct variant.
//
// The pass iterates to a fixed point (up to maxSpecializeIter rounds) to handle
// chains: if the body of a newly-created variant contains calls to other SPMD
// functions, those calls get specialization requests in the next round.
//
// Runs after Program.Build (all finishBody calls complete) so all SPMD passes
// (predication, propagation) have populated block annotations.
func spmdSpecializeFunctions(prog *Program) error {
	const maxSpecializeIter = 10
	for iter := 0; iter < maxSpecializeIter; iter++ {
		requests := spmdGatherRequests(prog)
		if len(requests) == 0 {
			return nil
		}
		anyChange := spmdMaterializeVariants(prog, requests)
		if !anyChange {
			return nil
		}
	}
	return fmt.Errorf("spmdSpecializeFunctions: did not reach fixed point after %d iterations", maxSpecializeIter)
}

// spmdSpecializeRequest records that fn should be specialized at laneCount.
type spmdSpecializeRequest struct {
	fn        *Function
	laneCount int
}

// spmdGatherRequests scans every function in prog for calls to SPMD functions
// (callees with lanes.Varying[T] parameters) and collects the set of
// (callee, laneCount) pairs that need specialization.
//
// Lane count is determined by (in priority order):
//  1. The call instruction's containing block's SPMDLaneCount (direct annotation).
//  2. Tracing each Varying-typed argument's defining chain back to an Alloc in
//     an annotated block (handles calls in post-loop "done" blocks where the
//     alloca was declared in the loop-entry block).
//
// Only lane counts > 0 are recorded.
func spmdGatherRequests(prog *Program) map[*Function]map[int]bool {
	requests := make(map[*Function]map[int]bool)

	for _, pkg := range prog.packages {
		for _, mem := range pkg.Members {
			if mem == nil {
				continue
			}
			fn, ok := mem.(*Function)
			if !ok || fn == nil || fn.Blocks == nil {
				continue
			}
			spmdGatherRequestsInFunc(fn, requests)
		}
	}
	return requests
}

// spmdGatherRequestsInFunc scans fn for calls to SPMD functions and records
// (callee, laneCount) requests into requests.
func spmdGatherRequestsInFunc(fn *Function, requests map[*Function]map[int]bool) {
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			call, ok := instr.(*Call)
			if !ok {
				continue
			}
			callee, ok := call.Call.Value.(*Function)
			if !ok {
				continue
			}
			if !spmdIsSPMDFunction(callee) {
				continue
			}
			// Skip already-specialized variants to avoid recursive renaming.
			// Variants are identified by having ".spmd" in their name (set
			// during the rename or clone step).
			if spmdIsAlreadySpecialized(callee.name) {
				continue
			}
			lc := spmdCallLaneCount(bb, call)
			if lc <= 0 {
				continue
			}
			if requests[callee] == nil {
				requests[callee] = make(map[int]bool)
			}
			requests[callee][lc] = true
		}
	}
}

// spmdIsAlreadySpecialized reports whether the function name indicates it is
// already a specialized variant (i.e. contains ".spmd" followed by digits).
func spmdIsAlreadySpecialized(name string) bool {
	const marker = ".spmd"
	idx := 0
	for {
		pos := -1
		for i := idx; i+len(marker) <= len(name); i++ {
			if name[i:i+len(marker)] == marker {
				pos = i
				break
			}
		}
		if pos < 0 {
			return false
		}
		// Check that what follows ".spmd" is all digits and non-empty.
		suffix := name[pos+len(marker):]
		if len(suffix) > 0 {
			allDigits := true
			for _, ch := range suffix {
				if ch < '0' || ch > '9' {
					allDigits = false
					break
				}
			}
			if allDigits {
				return true
			}
		}
		idx = pos + 1
	}
}

// spmdCallLaneCount determines the lane count for a call instruction.
//
// Priority:
//  1. Trace each Varying-typed argument back to its defining alloca's block
//     annotation. This is most precise because it follows the data flow of
//     each Varying value to the SPMD scope that produced it, regardless of
//     which block the call itself is in.
//  2. Fall back to the containing block's SPMDLaneCount. This handles calls
//     where the argument is not directly traceable to an alloca (e.g., the
//     Varying value is a function return or a phi with no annotated sources).
func spmdCallLaneCount(bb *BasicBlock, call *Call) int {
	// Priority 1: trace Varying arguments to their source alloca's lane count.
	for _, arg := range call.Call.Args {
		if !isLanesVaryingType(arg.Type()) {
			continue
		}
		lc := spmdTraceArgToAllocaLaneCount(arg)
		if lc > 0 {
			return lc
		}
	}
	// Priority 2: use the call block's annotation.
	if bb.SPMDLaneCount > 0 {
		return bb.SPMDLaneCount
	}
	return 0
}

// spmdTraceArgToAllocaLaneCount follows the value chain from arg (a
// lanes.Varying[T] value) back to its defining Alloc instruction and returns
// that alloca's block's SPMDLaneCount. Returns 0 if no annotated alloca is
// found within a short trace depth.
func spmdTraceArgToAllocaLaneCount(v Value) int {
	const maxDepth = 5
	return spmdTraceValue(v, maxDepth)
}

func spmdTraceValue(v Value, depth int) int {
	if depth <= 0 {
		return 0
	}
	switch x := v.(type) {
	case *Alloc:
		return x.Block().SPMDLaneCount
	case *UnOp:
		// *ptr load — trace the pointer back.
		if x.Op == token.MUL {
			return spmdTraceValue(x.X, depth-1)
		}
	case *FieldAddr:
		return spmdTraceValue(x.X, depth-1)
	case *IndexAddr:
		return spmdTraceValue(x.X, depth-1)
	case *Phi:
		// Trace first non-nil edge.
		for _, e := range x.Edges {
			if e != nil {
				lc := spmdTraceValue(e, depth-1)
				if lc > 0 {
					return lc
				}
			}
		}
	}
	return 0
}

// spmdIsSPMDFunction reports whether fn has lanes.Varying[T] parameters or
// results. Uses isLanesVaryingType which handles both *types.SPMDType (forked
// type checker in production) and *types.Named (standard importer in tests).
func spmdIsSPMDFunction(fn *Function) bool {
	if fn.Signature == nil {
		return false
	}
	sig := fn.Signature
	for i := 0; i < sig.Params().Len(); i++ {
		if isLanesVaryingType(sig.Params().At(i).Type()) {
			return true
		}
	}
	if sig.Results() != nil {
		for i := 0; i < sig.Results().Len(); i++ {
			if isLanesVaryingType(sig.Results().At(i).Type()) {
				return true
			}
		}
	}
	return false
}

// spmdMaterializeVariants creates (or renames) function variants for each
// (callee, lane count) pair in requests, and rewrites the corresponding call
// sites. Returns true if any change was made (so the caller can iterate).
func spmdMaterializeVariants(prog *Program, requests map[*Function]map[int]bool) bool {
	anyChange := false

	// Build a map from original function to its variants for call-site rewriting.
	// variants[orig][laneCount] = variant function.
	variants := make(map[*Function]map[int]*Function)

	for orig, lcSet := range requests {
		if orig.Pkg == nil || orig.Blocks == nil {
			continue
		}
		vmap := make(map[int]*Function)

		if len(lcSet) == 1 {
			// Single-lane shortcut: rename in place.
			var lc int
			for k := range lcSet {
				lc = k
			}
			newName := fmt.Sprintf("%s.spmd%d", orig.name, lc)
			// Remove old member, rename, add new member.
			delete(orig.Pkg.Members, orig.name)
			orig.name = newName
			// Clear the types.Func object reference so the sanity checker
			// does not compare orig.object.Name() ("sum") with the new
			// member key ("sum.spmd4"). With object=nil, Object() returns
			// nil and the sanity checker skips the name-equality check.
			orig.object = nil
			// Mark as synthetic and clear syntax so the sanity checker's
			// "Synthetic == '' iff has syntax" invariant is maintained.
			// A renamed SPMD variant is no longer a raw source function.
			orig.Synthetic = "SPMD specialization"
			orig.syntax = nil
			orig.Pkg.Members[newName] = orig
			// Annotate all blocks with the lane count.
			for _, bb := range orig.Blocks {
				if bb.SPMDLaneCount == 0 {
					bb.SPMDLaneCount = lc
				}
			}
			vmap[lc] = orig
			anyChange = true
		} else {
			// Multi-lane: clone once per lane count.
			for lc := range lcSet {
				clone := spmdCloneFunction(orig, lc)
				if clone == nil {
					// Non-cloneable function: skip specialization for this callee.
					continue
				}
				orig.Pkg.Members[clone.name] = clone
				vmap[lc] = clone
				anyChange = true
			}
			// Remove the original from Members once all variants are created.
			if len(vmap) > 0 {
				delete(orig.Pkg.Members, orig.name)
			}
		}
		variants[orig] = vmap
	}

	// Rewrite call sites across all functions.
	for _, pkg := range prog.packages {
		for _, mem := range pkg.Members {
			if mem == nil {
				continue
			}
			fn, ok := mem.(*Function)
			if !ok || fn == nil || fn.Blocks == nil {
				continue
			}
			spmdRewriteCallSites(fn, variants)
		}
	}

	return anyChange
}

// spmdCloneFunction deep-copies fn as a new variant specialized for laneCount.
// Returns nil if fn is not cloneable (contains instruction types that
// spmdCloneBlock does not handle, or has free variables).
//
// The clone is named "<fn.name>.spmd<laneCount>". Every block in the clone
// receives SPMDLaneCount=laneCount. The clone is registered in fn.Pkg.Members
// by the caller.
func spmdCloneFunction(orig *Function, laneCount int) *Function {
	// Reject non-cloneable bodies.
	for _, bb := range orig.Blocks {
		if !spmdBodyIsCloneable(bb) {
			return nil
		}
	}
	// Reject functions with free variables (closures).
	if len(orig.FreeVars) > 0 {
		return nil
	}

	cloneName := fmt.Sprintf("%s.spmd%d", orig.name, laneCount)

	clone := &Function{
		name:      cloneName,
		Signature: orig.Signature,
		Synthetic: "SPMD specialization",
		Pkg:       orig.Pkg,
		Prog:      orig.Prog,
		pos:       orig.pos,
	}

	// Value map: original value → clone value.
	// Pre-populate with parameter mappings.
	valueMap := make(map[Value]Value)

	// Clone parameters. Note: Parameter.Pos() comes from object.Pos(), but
	// synthetic clones share the signature so we leave object nil (pos = NoPos).
	clone.Params = make([]*Parameter, len(orig.Params))
	for i, p := range orig.Params {
		cp := &Parameter{
			name:   p.name,
			typ:    p.typ,
			parent: clone,
		}
		clone.Params[i] = cp
		valueMap[p] = cp
	}

	// Create clone blocks (no instructions yet).
	blockMap := make(map[*BasicBlock]*BasicBlock, len(orig.Blocks))
	clone.Blocks = make([]*BasicBlock, len(orig.Blocks))
	for i, bb := range orig.Blocks {
		cb := &BasicBlock{
			Index:         i,
			Comment:       bb.Comment,
			parent:        clone,
			SPMDLaneCount: laneCount,
		}
		blockMap[bb] = cb
		clone.Blocks[i] = cb
	}

	// Clone phis for each block (before cloning body instructions so that
	// phis are in valueMap when body instructions reference them).
	for _, bb := range orig.Blocks {
		cb := blockMap[bb]
		for _, instr := range bb.Instrs {
			phi, ok := instr.(*Phi)
			if !ok {
				break // phis are always first
			}
			cp := &Phi{
				Comment: phi.Comment,
				Edges:   make([]Value, len(phi.Edges)),
			}
			cp.setType(phi.Type())
			cp.setBlock(cb)
			cb.Instrs = append(cb.Instrs, cp)
			valueMap[phi] = cp
		}
	}

	// Clone body instructions (non-phi, non-terminator) for each block.
	for _, bb := range orig.Blocks {
		cb := blockMap[bb]
		// Collect allocs to populate Locals.
		spmdCloneBlock(clone, bb, cb, valueMap)
	}

	// Wire phis: translate edges using valueMap + blockMap.
	for _, bb := range orig.Blocks {
		cb := blockMap[bb]
		origPhis := spmdCollectPhis(bb)
		clonePhis := spmdCollectPhis(cb)
		for i, phi := range origPhis {
			cp := clonePhis[i]
			for j, edge := range phi.Edges {
				cv := spmdTranslateValue(edge, valueMap)
				cp.Edges[j] = cv
				spmdAddReferrer(cv, cp)
			}
		}
	}

	// Wire CFG edges and clone terminators.
	for _, bb := range orig.Blocks {
		cb := blockMap[bb]
		spmdCloneFuncTerminator(bb, cb, blockMap, valueMap)
	}

	// Populate Locals from Alloc instructions in clone blocks.
	for _, cb := range clone.Blocks {
		for _, instr := range cb.Instrs {
			if alloc, ok := instr.(*Alloc); ok && !alloc.Heap {
				clone.Locals = append(clone.Locals, alloc)
			}
		}
	}

	return clone
}

// spmdCollectPhis returns the prefix of phis from block's instruction list.
func spmdCollectPhis(bb *BasicBlock) []*Phi {
	var phis []*Phi
	for _, instr := range bb.Instrs {
		if phi, ok := instr.(*Phi); ok {
			phis = append(phis, phi)
		} else {
			break
		}
	}
	return phis
}

// spmdCloneFuncTerminator copies the terminator of srcBlock into dstBlock
// for the function-clone case, translating all successor references via
// blockMap and value references via valueMap.
func spmdCloneFuncTerminator(src, dst *BasicBlock, blockMap map[*BasicBlock]*BasicBlock, valueMap map[Value]Value) {
	if len(src.Instrs) == 0 {
		return
	}
	last := src.Instrs[len(src.Instrs)-1]
	switch term := last.(type) {
	case *Return:
		ret := &Return{
			Results: make([]Value, len(term.Results)),
		}
		ret.pos = term.pos
		for i, r := range term.Results {
			ret.Results[i] = spmdTranslateValue(r, valueMap)
			spmdAddReferrer(ret.Results[i], ret)
		}
		ret.setBlock(dst)
		dst.Instrs = append(dst.Instrs, ret)

	case *Jump:
		j := &Jump{}
		j.setBlock(dst)
		dst.Instrs = append(dst.Instrs, j)
		if len(src.Succs) > 0 {
			addEdge(dst, blockMap[src.Succs[0]])
		}

	case *If:
		ifInstr := &If{}
		ifInstr.Cond = spmdTranslateValue(term.Cond, valueMap)
		ifInstr.IsVarying = term.IsVarying
		ifInstr.setBlock(dst)
		dst.Instrs = append(dst.Instrs, ifInstr)
		spmdAddReferrer(ifInstr.Cond, ifInstr)
		if len(src.Succs) >= 2 {
			addEdge(dst, blockMap[src.Succs[0]])
			addEdge(dst, blockMap[src.Succs[1]])
		}

	case *Panic:
		pan := &Panic{}
		pan.X = spmdTranslateValue(term.X, valueMap)
		pan.pos = term.pos
		pan.setBlock(dst)
		dst.Instrs = append(dst.Instrs, pan)
		spmdAddReferrer(pan.X, pan)
	}

	// Wire predecessor list from successors.
	for _, succ := range dst.Succs {
		alreadyHas := false
		for _, pred := range succ.Preds {
			if pred == dst {
				alreadyHas = true
				break
			}
		}
		if !alreadyHas {
			succ.Preds = append(succ.Preds, dst)
		}
	}
}

// spmdRewriteCallSites rewrites call instructions in fn that target original
// SPMD functions to instead call the appropriate specialized variant based on
// the call site's lane count.
func spmdRewriteCallSites(fn *Function, variants map[*Function]map[int]*Function) {
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			call, ok := instr.(*Call)
			if !ok {
				continue
			}
			orig, ok := call.Call.Value.(*Function)
			if !ok {
				continue
			}
			vmap, hasVariants := variants[orig]
			if !hasVariants {
				continue
			}
			lc := spmdCallLaneCount(bb, call)
			if lc <= 0 {
				continue
			}
			variant, hasVariant := vmap[lc]
			if !hasVariant {
				continue
			}
			// Rewrite the call to point to the variant.
			// Remove this call as a referrer of the original function (if tracked).
			if refs := orig.Referrers(); refs != nil {
				*refs = spmdRemoveInstr(*refs, call)
			}
			call.Call.Value = variant
			// Add this call as a referrer of the variant (if tracked).
			spmdAddReferrer(variant, call)
		}
	}
}

// spmdRemoveInstr removes one occurrence of needle from haystack and returns
// the result. Used to clean up referrer lists when rewriting call values.
func spmdRemoveInstr(haystack []Instruction, needle Instruction) []Instruction {
	for i, v := range haystack {
		if v == needle {
			return append(haystack[:i], haystack[i+1:]...)
		}
	}
	return haystack
}


