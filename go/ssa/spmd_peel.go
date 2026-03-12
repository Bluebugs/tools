// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
)

// spmdTranslateValue looks up v in valueMap. If not found, returns v unchanged
// (the value is defined outside the cloned scope and can be used directly).
func spmdTranslateValue(v Value, valueMap map[Value]Value) Value {
	if mapped, ok := valueMap[v]; ok {
		return mapped
	}
	return v
}

// spmdCloneBlock copies the non-terminator, non-phi instructions from srcBlock
// into dstBlock, translating operand references via valueMap. For each
// value-producing instruction cloned, valueMap is updated so subsequent
// instructions within the same clone operation resolve correctly.
//
// Rules:
//   - DebugRef instructions are skipped (debug-only pseudo-instructions).
//   - Phi instructions are skipped; the caller creates and wires phis separately
//     because phi edges depend on the predecessor set of the target block.
//   - Jump and If terminators are skipped; the caller is responsible for wiring
//     block successors and adding the correct terminator.
//   - After buildReferrers has already run, every new instruction must call
//     spmdAddReferrer for each operand it references.
func spmdCloneBlock(fn *Function, srcBlock *BasicBlock, dstBlock *BasicBlock,
	valueMap map[Value]Value) {
	for _, instr := range srcBlock.Instrs {
		switch v := instr.(type) {
		case *DebugRef:
			// Skip debug pseudo-instructions; they carry source positions only.
			continue

		case *Phi:
			// Caller handles phi creation; phis depend on predecessor wiring.
			continue

		case *Jump:
			// Caller handles terminators.
			continue

		case *If:
			// Caller handles terminators.
			continue

		case *Return:
			// Caller handles terminators.
			continue

		case *Panic:
			// Caller handles terminators.
			continue

		case *BinOp:
			newInstr := &BinOp{Op: v.Op}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.Y = spmdTranslateValue(v.Y, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)
			spmdAddReferrer(newInstr.Y, newInstr)

		case *UnOp:
			newInstr := &UnOp{Op: v.Op, CommaOk: v.CommaOk}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)

		case *Store:
			newInstr := &Store{}
			newInstr.Addr = spmdTranslateValue(v.Addr, valueMap)
			newInstr.Val = spmdTranslateValue(v.Val, valueMap)
			newInstr.pos = v.pos
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			spmdAddReferrer(newInstr.Addr, newInstr)
			spmdAddReferrer(newInstr.Val, newInstr)

		case *SPMDStore:
			newInstr := &SPMDStore{Lanes: v.Lanes}
			newInstr.Addr = spmdTranslateValue(v.Addr, valueMap)
			newInstr.Val = spmdTranslateValue(v.Val, valueMap)
			newInstr.Mask = spmdTranslateValue(v.Mask, valueMap)
			newInstr.pos = v.pos
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			spmdAddReferrer(newInstr.Addr, newInstr)
			spmdAddReferrer(newInstr.Val, newInstr)
			spmdAddReferrer(newInstr.Mask, newInstr)

		case *SPMDLoad:
			newInstr := &SPMDLoad{Lanes: v.Lanes}
			newInstr.Addr = spmdTranslateValue(v.Addr, valueMap)
			newInstr.Mask = spmdTranslateValue(v.Mask, valueMap)
			newInstr.pos = v.pos
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.Addr, newInstr)
			spmdAddReferrer(newInstr.Mask, newInstr)

		case *SPMDSelect:
			newInstr := &SPMDSelect{Lanes: v.Lanes}
			newInstr.Mask = spmdTranslateValue(v.Mask, valueMap)
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.Y = spmdTranslateValue(v.Y, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.Mask, newInstr)
			spmdAddReferrer(newInstr.X, newInstr)
			spmdAddReferrer(newInstr.Y, newInstr)

		case *SPMDIndex:
			newInstr := &SPMDIndex{Lanes: v.Lanes, ElemType: v.ElemType}
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			// SPMDIndex has no operand references to register.

		case *SPMDExtractMask:
			clone := &SPMDExtractMask{
				X:     spmdTranslateValue(v.X, valueMap),
				Lanes: v.Lanes,
			}
			clone.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, clone)
			valueMap[v] = clone
			spmdAddReferrer(clone.X, clone)

		case *IndexAddr:
			newInstr := &IndexAddr{}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.Index = spmdTranslateValue(v.Index, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			// Preserve gather group annotation so TinyGo can use the same cache key
			// across both the main and tail peeled blocks.
			newInstr.SPMDGatherGroup = v.SPMDGatherGroup
			newInstr.SPMDGatherPos = v.SPMDGatherPos
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)
			spmdAddReferrer(newInstr.Index, newInstr)

		case *FieldAddr:
			newInstr := &FieldAddr{Field: v.Field}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)

		case *Convert:
			newInstr := &Convert{}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)

		case *ChangeType:
			newInstr := &ChangeType{}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)

		case *Call:
			newInstr := &Call{}
			newInstr.Call.Value = spmdTranslateValue(v.Call.Value, valueMap)
			newInstr.Call.Args = make([]Value, len(v.Call.Args))
			for i, arg := range v.Call.Args {
				newInstr.Call.Args[i] = spmdTranslateValue(arg, valueMap)
			}
			newInstr.Call.pos = v.Call.pos
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.Call.Value, newInstr)
			for _, arg := range newInstr.Call.Args {
				spmdAddReferrer(arg, newInstr)
			}

		case *Alloc:
			newInstr := &Alloc{Comment: v.Comment, Heap: v.Heap}
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			// Alloc has no input operands to register referrers for.

		case *SPMDVectorFromMemory:
			clone := &SPMDVectorFromMemory{
				Ptr:      spmdTranslateValue(v.Ptr, valueMap),
				Len:      spmdTranslateValue(v.Len, valueMap),
				ElemType: v.ElemType,
				Lanes:    v.Lanes,
			}
			clone.pos = v.pos
			clone.setType(v.Type())
			clone.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, clone)
			valueMap[v] = clone
			spmdAddReferrer(clone.Ptr, clone)
			spmdAddReferrer(clone.Len, clone)

		default:
			panic(fmt.Sprintf("spmdCloneBlock: unhandled instruction type %T", instr))
		}
	}
}

// peelSPMDLoops splits each SPMD loop in fn into a main (full-width) phase
// and a tail (masked) phase. Called from finishBody after predicateSPMD.
//
// For each loop with a resolved IterPhi and BoundValue, peelSPMDLoop creates:
//   - spmd.main.body: runs full laneCount-wide iterations
//   - spmd.tail.check: checks whether any tail iterations remain
//   - spmd.tail.body:  runs at most one iteration for the trailing elements
//
// Loops with laneCount <= 1 are skipped (no benefit from peeling).
// Loops without a resolved IterPhi or BoundValue are skipped.
func peelSPMDLoops(fn *Function) {
	for _, loop := range fn.SPMDLoops {
		if loop.IterPhi == nil || loop.BoundValue == nil {
			continue
		}
		peelSPMDLoop(fn, loop)
	}
}

// peelSPMDLoop transforms a single SPMD loop into main + tail form.
//
// Before peeling, the loop has the shape:
//
//	entry:
//	  If (0 < bound) → body, done
//
//	body (MergedBodyLoop=true: same block as loop):
//	  iter = phi [entry: 0, body: incr]
//	  ... body instructions ...
//	  incr = iter + 1
//	  If (incr < bound) → body, done
//
//	done:
//	  return
//
// After peeling:
//
//	entry:
//	  alignedBound = bound & ^(laneCount-1)
//	  hasMain = alignedBound > 0
//	  If hasMain → main.body, tail.check
//
//	main.body:
//	  mainIter = phi [entry: 0, main.body: mainIncr]
//	  ... cloned body instructions (iter → mainIter) ...
//	  mainIncr = mainIter + laneCount
//	  mainCond = mainIncr < alignedBound
//	  If mainCond → main.body, tail.check
//
//	tail.check:
//	  tailIter = phi [entry: 0, main.body: mainIncr]
//	  hasTail = tailIter < bound
//	  If hasTail → tail.body, done
//
//	tail.body:
//	  ... cloned body instructions (iter → tailIter) ...
//	  Jump → done
//
//	done:
//	  return
func peelSPMDLoop(fn *Function, loop *SPMDLoopInfo) {
	laneCount := loop.LaneCount
	if laneCount <= 1 {
		return
	}

	// Only handle merged body+loop blocks (rangeint form after block fusion).
	if !loop.MergedBodyLoop {
		return
	}

	body := loop.BodyBlock
	entry := loop.EntryBlock
	done := loop.DoneBlock

	// Only peel simple loops where the body block has exactly two successors:
	// itself (the self-loop back-edge) and done (the loop exit). After
	// predicateSPMD, a loop body with varying control flow (switch, if-else)
	// will have its terminator replaced with a Jump, producing a different
	// successor set. Those multi-block scopes are not yet supported.
	if len(body.Succs) != 2 || body.Succs[0] != body || body.Succs[1] != done {
		return
	}

	// Only peel loops whose body contains instruction types that spmdCloneBlock
	// knows how to copy. If the body contains an unsupported instruction (e.g.,
	// MakeInterface, Select, TypeAssert), skip peeling rather than panicking.
	if !spmdBodyIsCloneable(body) {
		return
	}

	// Use the IterPhi's type for all arithmetic. After the lift pass, the
	// IterPhi is always a typed integer (e.g., "int"), never "untyped int".
	// BoundValue may be untyped (e.g., a raw constant "8:untyped int"), so
	// we avoid using BoundValue.Type() for constant creation.
	intType := loop.IterPhi.Type()

	// Build a typed version of BoundValue for use in comparisons. If the
	// BoundValue is already typed (e.g., a variable), use it directly.
	// If it is an untyped constant, create a typed constant with the same
	// numeric value so all BinOp operands share the same type.
	typedBound := spmdTypedBound(loop.BoundValue, intType)

	// Constants for aligned-bound arithmetic:
	//   alignedBound = bound & ^(laneCount-1)
	// ^(laneCount-1) is -(laneCount) in two's complement integer notation.
	notMaskConst := NewConst(constant.MakeInt64(int64(^(laneCount-1))), intType)
	laneCountConst := NewConst(constant.MakeInt64(int64(laneCount)), intType)
	zeroConst := NewConst(constant.MakeInt64(0), intType)

	// Create three new blocks.
	mainBody := fn.newBasicBlock("spmd.main.body")
	tailCheck := fn.newBasicBlock("spmd.tail.check")
	tailBody := fn.newBasicBlock("spmd.tail.body")

	// --- Rewrite entry block ---
	// The entry block ends with:
	//   guardBinOp = 0 < bound   (result: bool)
	//   If(guardBinOp) → body, done
	//
	// We remove the If terminator, clear stale successor/predecessor edges,
	// and append the aligned-bound computation plus a new If.

	// Remove old If from entry.Instrs, also clean up the old If's referrer
	// in its condition operand so the sanity checker does not complain.
	n := len(entry.Instrs)
	if n == 0 {
		panic("peelSPMDLoop: entry block has no instructions")
	}
	oldIf, ok := entry.Instrs[n-1].(*If)
	if !ok {
		panic("peelSPMDLoop: entry block last instruction is not If")
	}

	// Extract the live bound from the guard condition before removing the If.
	// The guard has the form: guardCond = 0 < bound, where bound is the
	// post-lift live value. loop.BoundValue may be a stale load instruction
	// that was promoted away by the lift pass (its alloca was replaced with
	// the parameter or phi, but the SPMDLoopInfo field was not updated).
	// Using the guard's Y operand guarantees we reference the live value.
	if guardCond, ok2 := oldIf.Cond.(*BinOp); ok2 && guardCond.Op == token.LSS {
		typedBound = spmdTypedBound(guardCond.Y, intType)
	}

	// Remove old If from its Cond's referrer list.
	if refs := oldIf.Cond.Referrers(); refs != nil {
		*refs = spmdRemoveOneReferrer(*refs, oldIf)
	}
	entry.Instrs = entry.Instrs[:n-1]

	// Clear entry's old successors; we will add new ones via addEdge below.
	entry.Succs = entry.Succs[:0]

	// Remove entry as a predecessor of body and done (the old loop-entry edges).
	body.removePred(entry)
	done.removePred(entry)

	// Also remove body → done edge. The body block becomes unreachable after
	// peeling (no predecessor reaches it), but its If terminator still has
	// done as a successor. Removing body from done's predecessors here prevents
	// phi/pred count mismatches when new blocks (trampoline) are connected to
	// done later. Save each done-phi's body-edge value so we can wire the
	// correct values when adding the replacement predecessor.
	donePhiBodyEdges := map[*Phi]Value{}
	for _, instr := range done.phis() {
		phi := instr.(*Phi)
		for j, pred := range done.Preds {
			if pred == body {
				donePhiBodyEdges[phi] = phi.Edges[j]
			}
		}
	}
	done.removePred(body)

	// Append alignedBound = typedBound & ^(laneCount-1) into entry.
	alignedBound := &BinOp{Op: token.AND}
	alignedBound.X = typedBound
	alignedBound.Y = notMaskConst
	alignedBound.setType(intType)
	alignedBound.setBlock(entry)
	entry.Instrs = append(entry.Instrs, alignedBound)
	spmdAddReferrer(typedBound, alignedBound)
	spmdAddReferrer(notMaskConst, alignedBound)

	// Append hasMain = alignedBound > 0 into entry.
	hasMain := &BinOp{Op: token.GTR}
	hasMain.X = alignedBound
	hasMain.Y = zeroConst
	hasMain.setType(types.Typ[types.Bool])
	hasMain.setBlock(entry)
	entry.Instrs = append(entry.Instrs, hasMain)
	spmdAddReferrer(alignedBound, hasMain)
	spmdAddReferrer(zeroConst, hasMain)

	// Append new If terminator: If(hasMain) → mainBody, tailCheck.
	entryIf := &If{Cond: hasMain}
	entryIf.setBlock(entry)
	entry.Instrs = append(entry.Instrs, entryIf)
	spmdAddReferrer(hasMain, entryIf)

	// Wire entry → mainBody and entry → tailCheck.
	addEdge(entry, mainBody)
	addEdge(entry, tailCheck)

	// --- Populate main body ---
	// mainIter = phi [entry: 0, mainBody: mainIncr]
	// We insert the phi first with a nil back-edge placeholder and fill in
	// mainIncr after it is created.
	mainIterPhi := &Phi{Comment: "spmd.main.iter"}
	mainIterPhi.setType(intType)
	mainIterPhi.Edges = []Value{zeroConst, nil} // back-edge filled below
	spmdInsertPhiAtFront(mainBody, mainIterPhi)
	spmdAddReferrer(zeroConst, mainIterPhi)

	// Create accumulator phis in mainBody BEFORE cloning so the valueMap
	// maps each original accumulator phi to its mainBody counterpart. The
	// back-edge for each mainAccPhi is filled in after spmdCloneBlock, once
	// the cloned back-edge value is available.
	//
	// spmdInsertPhiAtFront inserts at position 0, so multiple accumulator
	// phis end up in reverse declaration order; this is acceptable because
	// edge ordering only needs to match Preds ordering, which it does.
	mainAccPhis := make([]*Phi, len(loop.Accumulators))
	mainValueMap := map[Value]Value{loop.IterPhi: mainIterPhi}
	for i, acc := range loop.Accumulators {
		mainAccPhi := &Phi{Comment: "spmd.main.acc"}
		mainAccPhi.setType(acc.Phi.Type())
		mainAccPhi.Edges = []Value{acc.InitValue, nil} // back-edge filled after clone
		spmdInsertPhiAtFront(mainBody, mainAccPhi)
		spmdAddReferrer(acc.InitValue, mainAccPhi)
		mainValueMap[acc.Phi] = mainAccPhi
		mainAccPhis[i] = mainAccPhi
	}

	// Clone body instructions using mainValueMap so that references to the
	// original IterPhi and accumulator phis resolve to their mainBody copies.
	spmdCloneBlock(fn, body, mainBody, mainValueMap)

	// mainIncr = mainIterPhi + laneCount  (advance by a full SIMD width).
	mainIncr := &BinOp{Op: token.ADD}
	mainIncr.X = mainIterPhi
	mainIncr.Y = laneCountConst
	mainIncr.setType(intType)
	mainIncr.setBlock(mainBody)
	mainBody.Instrs = append(mainBody.Instrs, mainIncr)
	spmdAddReferrer(mainIterPhi, mainIncr)
	spmdAddReferrer(laneCountConst, mainIncr)

	// Fill in the back-edge of mainIterPhi now that mainIncr exists.
	mainIterPhi.Edges[1] = mainIncr
	spmdAddReferrer(mainIncr, mainIterPhi)

	// Fill in the back-edges of accumulator phis now that the cloned
	// back-edge values are available in mainValueMap.
	for i, acc := range loop.Accumulators {
		backVal := spmdTranslateValue(acc.BackValue, mainValueMap)
		mainAccPhis[i].Edges[1] = backVal
		spmdAddReferrer(backVal, mainAccPhis[i])
	}

	// mainCond = mainIncr < alignedBound.
	mainCond := &BinOp{Op: token.LSS}
	mainCond.X = mainIncr
	mainCond.Y = alignedBound
	mainCond.setType(types.Typ[types.Bool])
	mainCond.setBlock(mainBody)
	mainBody.Instrs = append(mainBody.Instrs, mainCond)
	spmdAddReferrer(mainIncr, mainCond)
	spmdAddReferrer(alignedBound, mainCond)

	// If(mainCond) → mainBody, tailCheck.
	mainIf := &If{Cond: mainCond}
	mainIf.setBlock(mainBody)
	mainBody.Instrs = append(mainBody.Instrs, mainIf)
	spmdAddReferrer(mainCond, mainIf)

	// Wire: mainBody self-loop and mainBody → tailCheck.
	// After these two addEdge calls, mainBody.Preds = [entry, mainBody],
	// which matches mainIterPhi.Edges = [zeroConst, mainIncr].
	addEdge(mainBody, mainBody)
	addEdge(mainBody, tailCheck)

	// --- Determine exit target for tailCheck and tailBody ---
	// When the loop has accumulators, the done block may have phis that consume
	// the final accumulator value. After peeling, done can be reached from two
	// paths (tailCheck when no tail, tailBody after tail executes) and those
	// paths carry different accumulator values. A trampoline block merges them.
	// For loops without accumulators, both tailCheck and tailBody branch directly
	// to done; exitBlock is set to done in that case.
	exitBlock := done
	var trampoline *BasicBlock
	if len(loop.Accumulators) > 0 {
		trampoline = fn.newBasicBlock("spmd.trampoline")
		exitBlock = trampoline
		loop.TrampolineBlock = trampoline
	}

	// --- Populate tail check ---
	// tailIter = phi [entry: 0, mainBody: mainIncr]
	// tailCheck.Preds will be [entry, mainBody] because addEdge(entry, tailCheck)
	// was called before addEdge(mainBody, tailCheck) above.
	tailIterPhi := &Phi{Comment: "spmd.tail.iter"}
	tailIterPhi.setType(intType)
	tailIterPhi.Edges = []Value{zeroConst, mainIncr}
	spmdInsertPhiAtFront(tailCheck, tailIterPhi)
	spmdAddReferrer(zeroConst, tailIterPhi)
	spmdAddReferrer(mainIncr, tailIterPhi)

	// Create accumulator phis in tailCheck, parallel to the iter phi.
	// Each tailAccPhi receives the initial value on the entry edge and the
	// main body's final accumulator result on the mainBody edge.
	// tailValueMap maps the original acc.Phi to the tailAccPhi so the
	// tail body clone resolves accumulator references correctly.
	tailAccPhis := make([]*Phi, len(loop.Accumulators))
	tailValueMap := map[Value]Value{loop.IterPhi: tailIterPhi}
	for i, acc := range loop.Accumulators {
		mainAccResult := spmdTranslateValue(acc.BackValue, mainValueMap)
		tailAccPhi := &Phi{Comment: "spmd.tail.acc"}
		tailAccPhi.setType(acc.Phi.Type())
		// Edges align with tailCheck.Preds = [entry, mainBody].
		tailAccPhi.Edges = []Value{acc.InitValue, mainAccResult}
		spmdInsertPhiAtFront(tailCheck, tailAccPhi)
		spmdAddReferrer(acc.InitValue, tailAccPhi)
		spmdAddReferrer(mainAccResult, tailAccPhi)
		tailValueMap[acc.Phi] = tailAccPhi
		tailAccPhis[i] = tailAccPhi
	}

	// hasTail = tailIterPhi < typedBound.
	hasTail := &BinOp{Op: token.LSS}
	hasTail.X = tailIterPhi
	hasTail.Y = typedBound
	hasTail.setType(types.Typ[types.Bool])
	hasTail.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, hasTail)
	spmdAddReferrer(tailIterPhi, hasTail)
	spmdAddReferrer(typedBound, hasTail)

	// If(hasTail) → tailBody, exitBlock (done or trampoline).
	tailCheckIf := &If{Cond: hasTail}
	tailCheckIf.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, tailCheckIf)
	spmdAddReferrer(hasTail, tailCheckIf)

	// Wire tailCheck → tailBody and tailCheck → exitBlock.
	addEdge(tailCheck, tailBody)
	addEdge(tailCheck, exitBlock)

	// --- Populate tail body ---
	spmdCloneBlock(fn, body, tailBody, tailValueMap)

	// Jump → exitBlock (done or trampoline).
	tailJump := &Jump{}
	tailJump.setBlock(tailBody)
	tailBody.Instrs = append(tailBody.Instrs, tailJump)

	// Wire tailBody → exitBlock.
	addEdge(tailBody, exitBlock)

	// --- Build trampoline (only when accumulators are present) ---
	// The trampoline merges the accumulator value from the two paths that
	// reach it: tailCheck (no tail iterations) and tailBody (after tail).
	// It then jumps unconditionally to done.
	//
	// trampoline.Preds = [tailCheck, tailBody] in the order addEdge was called,
	// so mergePhi.Edges = [tailAccPhi, tailAccResult] matches that order.
	if trampoline != nil {
		// Build a map from original accumulator values → merge phi so we can
		// wire done-block phis correctly below. The done-block phi may
		// reference either acc.Phi (the loop-carried phi) or acc.BackValue
		// (the updated value, e.g. accPhi + something), so map both.
		accMergeMap := make(map[Value]Value, len(loop.Accumulators)*2)
		for i, acc := range loop.Accumulators {
			tailAccResult := spmdTranslateValue(acc.BackValue, tailValueMap)
			mergePhi := &Phi{Comment: "spmd.acc.merge"}
			mergePhi.setType(acc.Phi.Type())
			// Edges align with trampoline.Preds = [tailCheck, tailBody].
			mergePhi.Edges = []Value{tailAccPhis[i], tailAccResult}
			spmdInsertPhiAtFront(trampoline, mergePhi)
			spmdAddReferrer(tailAccPhis[i], mergePhi)
			spmdAddReferrer(tailAccResult, mergePhi)

			accMergeMap[acc.Phi] = mergePhi
			accMergeMap[acc.BackValue] = mergePhi

			// Redirect post-loop uses of the original accumulator phi to the
			// merged result. We deliberately skip uses that are in the original
			// body block: those instructions are about to become unreachable
			// (deleteUnreachableBlocks runs after peeling), and updating them
			// would add dead instructions to mergePhi.Referrers(), which
			// causes the post-build sanity checker to fail.
			spmdReplaceAccUses(acc.Phi, mergePhi, body)
		}

		// Add unconditional Jump → done and wire the edge.
		trampJump := &Jump{}
		trampJump.setBlock(trampoline)
		trampoline.Instrs = append(trampoline.Instrs, trampJump)
		addEdge(trampoline, done)

		// Wire done-block phis for the new trampoline predecessor. We removed
		// body from done's preds earlier (body is unreachable), so done's phis
		// currently have 0 edges. The trampoline is now the sole predecessor.
		for _, instr := range done.phis() {
			phi := instr.(*Phi)
			bodyEdge := donePhiBodyEdges[phi]
			if mergeVal, ok := accMergeMap[bodyEdge]; ok {
				phi.Edges = append(phi.Edges, mergeVal)
				spmdAddReferrer(mergeVal, phi)
				continue
			}
			// Non-accumulator phi: carry the original body-edge value.
			phi.Edges = append(phi.Edges, bodyEdge)
		}
	}

	// --- Update loop metadata ---
	loop.IsPeeled = true
	loop.MainBodyBlock = mainBody
	loop.TailCheckBlock = tailCheck
	loop.TailBodyBlock = tailBody
	loop.AlignedBound = alignedBound
	loop.MainIterPhi = mainIterPhi
	loop.TailIterPhi = tailIterPhi
	// Update BoundValue to the live (post-lift) value derived from the guard
	// condition. The original BoundValue may be a stale load instruction that
	// the lift pass promoted away; typedBound holds the current live value.
	loop.BoundValue = typedBound
	// TrampolineBlock is set above when accumulators are present, nil otherwise.
}

// spmdReplaceAccUses replaces uses of the original accumulator phi (oldPhi)
// with newVal (the merge phi in the trampoline block), but only for instructions
// that are NOT in excludeBlock (the original loop body block). Instructions in
// excludeBlock are about to become unreachable and must not be added to
// newVal.Referrers(); doing so would cause the post-build sanity checker to
// report a stale referrer error.
func spmdReplaceAccUses(oldPhi *Phi, newVal Value, excludeBlock *BasicBlock) {
	pxrefs := oldPhi.Referrers()
	if pxrefs == nil {
		return
	}
	pyrefs := newVal.Referrers()
	var remaining []Instruction
	for _, instr := range *pxrefs {
		if instr.Block() == excludeBlock {
			// Keep this referrer on the old phi; it lives in the dead block.
			remaining = append(remaining, instr)
			continue
		}
		// Update instr's operand pointer from oldPhi to newVal.
		var rands []*Value
		rands = instr.Operands(rands[:0])
		for _, rand := range rands {
			if rand != nil && *rand == Value(oldPhi) {
				*rand = newVal
			}
		}
		if pyrefs != nil {
			*pyrefs = append(*pyrefs, instr)
		}
	}
	*pxrefs = remaining
}

// spmdBodyIsCloneable reports whether all non-phi, non-terminator instructions
// in block can be copied by spmdCloneBlock. If any instruction has a type not
// listed in spmdCloneBlock's switch, peeling must be skipped.
func spmdBodyIsCloneable(block *BasicBlock) bool {
	for _, instr := range block.Instrs {
		switch instr.(type) {
		case *DebugRef, *Phi, *Jump, *If, *Return, *Panic:
			// Skipped by spmdCloneBlock; always OK.
		case *BinOp, *UnOp, *Store, *SPMDStore, *SPMDLoad, *SPMDSelect, *SPMDIndex,
			*SPMDExtractMask, *IndexAddr, *FieldAddr, *Convert, *ChangeType, *Call, *Alloc:
			// Handled by spmdCloneBlock; OK.
		default:
			return false
		}
	}
	return true
}

// spmdTypedBound returns a typed version of bound for use in integer arithmetic.
// If bound already has type intType (or is compatible), it is returned as-is.
// If bound is an untyped constant, a new typed Const with the same numeric value
// is returned so that BinOp operands share a consistent type.
func spmdTypedBound(bound Value, intType types.Type) Value {
	if !isUntyped(bound.Type()) {
		return bound
	}
	// bound is an untyped constant: extract its numeric value and retype it.
	c, ok := bound.(*Const)
	if !ok {
		// Non-constant untyped values should not occur for loop bounds, but
		// fall back to the original value to avoid a panic.
		return bound
	}
	return NewConst(c.Value, intType)
}
