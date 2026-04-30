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
			// Non-heap allocas must appear in fn.Locals so the sanity
			// checker and lift pass see them as frame-allocated variables.
			if !newInstr.Heap {
				fn.Locals = append(fn.Locals, newInstr)
			}

		case *MakeInterface:
			newInstr := &MakeInterface{SPMDLanes: v.SPMDLanes}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			if v.SPMDMask != nil {
				newInstr.SPMDMask = spmdTranslateValue(v.SPMDMask, valueMap)
			}
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
			dstBlock.Instrs = append(dstBlock.Instrs, newInstr)
			valueMap[v] = newInstr
			spmdAddReferrer(newInstr.X, newInstr)
			if newInstr.SPMDMask != nil {
				spmdAddReferrer(newInstr.SPMDMask, newInstr)
			}

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
		// Skip peeling for scalar fallback (laneCount=1): no main/tail split
		// is needed when the loop processes one element at a time.
		if loop.LaneCount <= 1 {
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

	body := loop.BodyBlock
	entry := loop.EntryBlock
	done := loop.DoneBlock

	if loop.MergedBodyLoop {
		// Merged rangeint form: body block is its own loop header.
		// Classify the loop body shape. shapeSimple is the existing fast path.
		// shapeUniformExit extends peeling to loops whose body ends with a
		// uniform If (e.g., "if reduce.Any(found) { return true }") before
		// the back-edge.
		shape, earlyExitBlock, chainBlocks, backEdgeBlock := spmdClassifyLoopBody(body, done)
		switch shape {
		case shapeSimple:
			// Existing fast path: body.Succs == [body, done].
			if !spmdBodyIsCloneable(body) {
				return
			}
			peelSPMDLoopSimple(fn, loop, laneCount, body, entry, done)
			return
		case shapeUniformExit:
			// Multi-block body with a uniform early-exit before the back-edge.
			if !spmdBodyIsCloneable(body) {
				return
			}
			for _, cb := range chainBlocks {
				if !spmdBodyIsCloneable(cb) {
					return
				}
			}
			if !spmdBodyIsCloneable(backEdgeBlock) {
				return
			}
			if !spmdBodyIsCloneable(earlyExitBlock) {
				return
			}
			peelSPMDLoopUniformExit(fn, loop, laneCount, body, entry, done,
				earlyExitBlock, chainBlocks, backEdgeBlock)
			return
		default:
			return
		}
	}

	// Non-merged rangeindex form: loopBlock and bodyBlock are separate.
	// The loop structure is:
	//   entry → loopBlock (unconditionally)
	//   loopBlock: iter phi, incr, cond, If → body, done
	//   body: SPMD ops, uniform If → earlyExit, loopBlock  (shapeRangeIndexUniformExit)
	//     OR body: SPMD ops, Jump → loopBlock              (shapeRangeIndexSimple, not yet handled)
	//   earlyExit: Return
	//   done: Return
	loopBlock := loop.LoopBlock
	if loopBlock == nil || body == nil || done == nil {
		return
	}
	if !spmdBodyIsCloneable(body) {
		return
	}
	if !spmdBodyIsCloneable(loopBlock) {
		return
	}

	// Check for the uniform-exit shape: body ends with a non-varying If
	// where one successor is a dedicated exit block (Return or Jump→done),
	// and the other is loopBlock (the continuation back to the loop header).
	var rangeIdxExitBlock *BasicBlock
	if len(body.Succs) == 2 {
		bodyIf, ok := body.Instrs[len(body.Instrs)-1].(*If)
		if ok && !bodyIf.IsVarying {
			for _, succ := range [2]*BasicBlock{body.Succs[0], body.Succs[1]} {
				// succ must not be done or loopBlock itself.
				if succ == done || succ == loopBlock {
					continue
				}
				if spmdIsExitBlock(succ, done) {
					other := body.Succs[0]
					if succ == body.Succs[0] {
						other = body.Succs[1]
					}
					if other == loopBlock {
						rangeIdxExitBlock = succ
					}
				}
			}
		}
	}

	if rangeIdxExitBlock == nil {
		// Neither shapeSimple nor shapeUniformExit for rangeindex: bail.
		return
	}
	if !spmdBodyIsCloneable(rangeIdxExitBlock) {
		return
	}

	peelSPMDLoopRangeIndexUniformExit(fn, loop, laneCount,
		entry, loopBlock, body, done, rangeIdxExitBlock)
}

// peelSPMDLoopSimple is the existing fast-path peeler for loops whose body
// block is its own back-edge (Succs[0]==body, Succs[1]==done). The full
// description is in the peelSPMDLoop doc comment above.
func peelSPMDLoopSimple(fn *Function, loop *SPMDLoopInfo, laneCount int,
	body, entry, done *BasicBlock) {

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
		panic("peelSPMDLoopSimple: entry block has no instructions")
	}
	oldIf, ok := entry.Instrs[n-1].(*If)
	if !ok {
		panic("peelSPMDLoopSimple: entry block last instruction is not If")
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

	// Save done-phi entry-edges BEFORE removing entry from done.Preds so
	// we can wire the "no-main-iterations" path after peeling.
	donePhiEntryEdges := map[*Phi]Value{}
	for _, instr := range done.phis() {
		phi := instr.(*Phi)
		for j, pred := range done.Preds {
			if pred == entry {
				donePhiEntryEdges[phi] = phi.Edges[j]
			}
		}
	}

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
	// When the loop has accumulators, or when done has phis (values computed in
	// the loop body and consumed after it), a trampoline block is needed to merge
	// the two exit paths (tailCheck = no tail, tailBody = after tail) before
	// jumping to done. Without a trampoline, done would have two predecessors but
	// its phis would have no edges — a malformed SSA state.
	tailExitTarget := done
	var trampoline *BasicBlock
	if len(loop.Accumulators) > 0 || len(done.phis()) > 0 {
		trampoline = fn.newBasicBlock("spmd.trampoline")
		tailExitTarget = trampoline
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

	// If(hasTail) → tailBody, tailExitTarget (done or trampoline).
	tailCheckIf := &If{Cond: hasTail}
	tailCheckIf.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, tailCheckIf)
	spmdAddReferrer(hasTail, tailCheckIf)

	// Wire tailCheck → tailBody and tailCheck → tailExitTarget.
	addEdge(tailCheck, tailBody)
	addEdge(tailCheck, tailExitTarget)

	// --- Populate tail body ---
	spmdCloneBlock(fn, body, tailBody, tailValueMap)

	// Jump → tailExitTarget (done or trampoline).
	tailJump := &Jump{}
	tailJump.setBlock(tailBody)
	tailBody.Instrs = append(tailBody.Instrs, tailJump)

	// Wire tailBody → tailExitTarget.
	addEdge(tailBody, tailExitTarget)

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
				// Body-block accumulator: the merge phi in the trampoline
				// already holds the correct merged value.
				phi.Edges = append(phi.Edges, mergeVal)
				spmdAddReferrer(mergeVal, phi)
				continue
			}
			// Done-block phi (not a body-block accumulator): the phi lives in
			// done rather than body, so it's not in loop.Accumulators. We
			// still need to merge the two exit paths (tailCheck, tailBody).
			//
			// Create a tailCheck phi that carries:
			//   - from entry: the original entry-edge value (no iterations ran)
			//   - from mainBody: the translated back-value (after main iterations)
			// Then create a trampoline merge phi with:
			//   - from tailCheck: the tailCheck phi
			//   - from tailBody: the translated back-value for the tail iteration
			entryEdge := donePhiEntryEdges[phi]
			if entryEdge == nil {
				// Edge not found (shouldn't happen in well-formed SSA); use bodyEdge.
				entryEdge = bodyEdge
			}
			mainBackEdge := spmdTranslateValue(bodyEdge, mainValueMap)
			tailCheckPhi := &Phi{Comment: "spmd.done.acc"}
			tailCheckPhi.setType(phi.Type())
			// tailCheck.Preds = [entry, mainBody] (see addEdge calls above).
			tailCheckPhi.Edges = []Value{entryEdge, mainBackEdge}
			spmdInsertPhiAtFront(tailCheck, tailCheckPhi)
			spmdAddReferrer(entryEdge, tailCheckPhi)
			spmdAddReferrer(mainBackEdge, tailCheckPhi)

			tailBodyEdge := spmdTranslateValue(bodyEdge, tailValueMap)
			mergePhi := &Phi{Comment: "spmd.done.merge"}
			mergePhi.setType(phi.Type())
			// trampoline.Preds = [tailCheck, tailBody].
			mergePhi.Edges = []Value{tailCheckPhi, tailBodyEdge}
			spmdInsertPhiAtFront(trampoline, mergePhi)
			spmdAddReferrer(tailCheckPhi, mergePhi)
			spmdAddReferrer(tailBodyEdge, mergePhi)

			phi.Edges = append(phi.Edges, mergePhi)
			spmdAddReferrer(mergePhi, phi)
		}
	}

	// --- Update loop metadata ---
	loop.IsPeeled = true
	loop.MainBodyBlock = mainBody
	loop.TailCheckBlock = tailCheck
	loop.TailBodyBlock = tailBody
	loop.BackEdgeBlock = body // shapeSimple: the body is its own back-edge
	loop.AlignedBound = alignedBound
	loop.MainIterPhi = mainIterPhi
	loop.TailIterPhi = tailIterPhi
	// Update BoundValue to the live (post-lift) value derived from the guard
	// condition. The original BoundValue may be a stale load instruction that
	// the lift pass promoted away; typedBound holds the current live value.
	loop.BoundValue = typedBound
	// TrampolineBlock is set above when accumulators are present, nil otherwise.
}

// peelSPMDLoopUniformExit peels a loop whose body ends with a uniform (non-
// varying) If that provides an early exit before the loop back-edge. After
// peeling, both the main and tail phases contain clones of the full block chain
// (body + chain blocks + back-edge block), each with its own clone of the early-
// exit block. The return/jump in each exit clone is independently translatable
// through the per-clone valueMap.
//
// The concrete SSA shape handled here (shapeUniformExit):
//
//	entry:      If (0 < bound) → body, done
//
//	body:       phi iter [entry: 0, backEdge: incr]
//	            ... body instructions ...
//	            If uniformCond → earlyExit, chainStart
//
//	earlyExit:  Return <val>               (or Jump → done)
//
//	chainBlocks: zero or more straight-line blocks
//
//	backEdge:   ... incr computations ...
//	            If (incr < bound) → body, done
//
//	done:       Return <val>
//
// After peeling:
//
//	entry:      alignedBound = bound & ^(lc-1)
//	            If (alignedBound > 0) → mainBody, tailCheck
//
//	mainBody:   phi mainIter [entry: 0, mainBackEdge: mainIncr]
//	            ... cloned body ... cloned chain ...
//	            If clonedCond → mainEarlyExit, mainBackEdge
//
//	mainEarlyExit: Return <translated val>
//
//	mainBackEdge: mainIncr = mainIter + laneCount
//	              If (mainIncr < alignedBound) → mainBody, tailCheck
//
//	tailCheck:  phi tailIter [entry: 0, mainBackEdge: mainIncr]
//	            If (tailIter < bound) → tailBody, done
//
//	tailBody:   ... cloned body ... cloned chain ...
//	            If clonedCond → tailEarlyExit, tailBackEdge
//
//	tailEarlyExit: Return <translated val>
//
//	tailBackEdge: Jump → done
//
//	done:       Return false
func peelSPMDLoopUniformExit(fn *Function, loop *SPMDLoopInfo, laneCount int,
	body, entry, done, earlyExitBlock *BasicBlock,
	chainBlocks []*BasicBlock, backEdgeBlock *BasicBlock) {

	intType := loop.IterPhi.Type()
	typedBound := spmdTypedBound(loop.BoundValue, intType)

	// Extract live bound from entry guard (same logic as peelSPMDLoopSimple).
	n := len(entry.Instrs)
	if n == 0 {
		panic("peelSPMDLoopUniformExit: entry block has no instructions")
	}
	oldEntryIf, ok := entry.Instrs[n-1].(*If)
	if !ok {
		panic("peelSPMDLoopUniformExit: entry block last instruction is not If")
	}
	if guardCond, ok2 := oldEntryIf.Cond.(*BinOp); ok2 && guardCond.Op == token.LSS {
		typedBound = spmdTypedBound(guardCond.Y, intType)
	}

	notMaskConst := NewConst(constant.MakeInt64(int64(^(laneCount-1))), intType)
	laneCountConst := NewConst(constant.MakeInt64(int64(laneCount)), intType)
	zeroConst := NewConst(constant.MakeInt64(0), intType)

	// Create new blocks for main and tail phases.
	mainBody := fn.newBasicBlock("spmd.main.body")
	mainEarlyExit := fn.newBasicBlock("spmd.main.exit")
	mainBackEdge := fn.newBasicBlock("spmd.main.backedge")
	tailCheck := fn.newBasicBlock("spmd.tail.check")
	tailBody := fn.newBasicBlock("spmd.tail.body")
	tailEarlyExit := fn.newBasicBlock("spmd.tail.exit")
	tailBackEdge := fn.newBasicBlock("spmd.tail.backedge")

	// Also create cloned chain blocks for main and tail phases.
	// chainBlocks may be empty (direct body → backEdge connection).
	mainChain := make([]*BasicBlock, len(chainBlocks))
	tailChain := make([]*BasicBlock, len(chainBlocks))
	for i, cb := range chainBlocks {
		mainChain[i] = fn.newBasicBlock("spmd.main.chain." + cb.Comment)
		tailChain[i] = fn.newBasicBlock("spmd.tail.chain." + cb.Comment)
	}

	// --- Rewrite entry block ---
	if refs := oldEntryIf.Cond.Referrers(); refs != nil {
		*refs = spmdRemoveOneReferrer(*refs, oldEntryIf)
	}
	entry.Instrs = entry.Instrs[:n-1]
	entry.Succs = entry.Succs[:0]

	// Snapshot done-phi entry edges before rewiring.
	donePhiEntryEdges := map[*Phi]Value{}
	for _, instr := range done.phis() {
		phi := instr.(*Phi)
		for j, pred := range done.Preds {
			if pred == entry {
				donePhiEntryEdges[phi] = phi.Edges[j]
			}
		}
	}

	// Remove entry→body and entry→done edges; also remove backEdge→done.
	// In shapeUniformExit the done block is reached from backEdgeBlock (not body).
	body.removePred(entry)
	done.removePred(entry)

	// Save backEdge→done phi values before removing backEdge from done.Preds.
	// These values flow from the back-edge block, not the body block.
	donePhiBackEdgeEdges := map[*Phi]Value{}
	for _, instr := range done.phis() {
		phi := instr.(*Phi)
		for j, pred := range done.Preds {
			if pred == backEdgeBlock {
				donePhiBackEdgeEdges[phi] = phi.Edges[j]
			}
		}
	}
	done.removePred(backEdgeBlock)

	// Remove earlyExitBlock from done.Preds if it was a direct Jump→done.
	// (When earlyExitBlock has Return terminator this is a no-op.)
	if len(earlyExitBlock.Succs) == 1 && earlyExitBlock.Succs[0] == done {
		done.removePred(earlyExitBlock)
	}

	// Build aligned-bound computation in entry.
	alignedBound := &BinOp{Op: token.AND}
	alignedBound.X = typedBound
	alignedBound.Y = notMaskConst
	alignedBound.setType(intType)
	alignedBound.setBlock(entry)
	entry.Instrs = append(entry.Instrs, alignedBound)
	spmdAddReferrer(typedBound, alignedBound)
	spmdAddReferrer(notMaskConst, alignedBound)

	hasMain := &BinOp{Op: token.GTR}
	hasMain.X = alignedBound
	hasMain.Y = zeroConst
	hasMain.setType(types.Typ[types.Bool])
	hasMain.setBlock(entry)
	entry.Instrs = append(entry.Instrs, hasMain)
	spmdAddReferrer(alignedBound, hasMain)
	spmdAddReferrer(zeroConst, hasMain)

	entryIf := &If{Cond: hasMain}
	entryIf.setBlock(entry)
	entry.Instrs = append(entry.Instrs, entryIf)
	spmdAddReferrer(hasMain, entryIf)
	addEdge(entry, mainBody)
	addEdge(entry, tailCheck)

	// --- Populate main body ---
	// mainIterPhi: phi [entry: 0, mainBackEdge: mainIncr]
	// Preds order will be [entry, mainBackEdge] matching edge order.
	mainIterPhi := &Phi{Comment: "spmd.main.iter"}
	mainIterPhi.setType(intType)
	mainIterPhi.Edges = []Value{zeroConst, nil} // back-edge filled after mainBackEdge is built
	spmdInsertPhiAtFront(mainBody, mainIterPhi)
	spmdAddReferrer(zeroConst, mainIterPhi)

	// Accumulator phis in mainBody (same pattern as peelSPMDLoopSimple).
	mainAccPhis := make([]*Phi, len(loop.Accumulators))
	mainValueMap := map[Value]Value{loop.IterPhi: mainIterPhi}
	for i, acc := range loop.Accumulators {
		mainAccPhi := &Phi{Comment: "spmd.main.acc"}
		mainAccPhi.setType(acc.Phi.Type())
		mainAccPhi.Edges = []Value{acc.InitValue, nil}
		spmdInsertPhiAtFront(mainBody, mainAccPhi)
		spmdAddReferrer(acc.InitValue, mainAccPhi)
		mainValueMap[acc.Phi] = mainAccPhi
		mainAccPhis[i] = mainAccPhi
	}

	// Clone body instructions into mainBody.
	spmdCloneBlock(fn, body, mainBody, mainValueMap)

	// Wire the main body's cloned uniform-exit If:
	//   Succs[0] (true  branch) → mainEarlyExit
	//   Succs[1] (false branch) → first chain block or mainBackEdge
	mainBodyCont := spmdFirstChainOrBackEdge(mainChain, mainBackEdge)
	addEdge(mainBody, mainEarlyExit)
	addEdge(mainBody, mainBodyCont)
	// Restore the cloned If terminator targeting the new blocks. spmdCloneBlock
	// skips If terminators; we add the cloned If here using the translated cond.
	bodyTermIf := body.Instrs[len(body.Instrs)-1].(*If)
	mainBodyIf := &If{Cond: spmdTranslateValue(bodyTermIf.Cond, mainValueMap)}
	mainBodyIf.setBlock(mainBody)
	mainBody.Instrs = append(mainBody.Instrs, mainBodyIf)
	spmdAddReferrer(mainBodyIf.Cond, mainBodyIf)

	// --- Populate main chain blocks ---
	spmdCloneChainBlocks(fn, chainBlocks, mainChain, mainBackEdge, mainValueMap)

	// --- Populate main back-edge block ---
	spmdCloneBlock(fn, backEdgeBlock, mainBackEdge, mainValueMap)

	// mainIncr = mainIterPhi + laneCount.
	mainIncr := &BinOp{Op: token.ADD}
	mainIncr.X = mainIterPhi
	mainIncr.Y = laneCountConst
	mainIncr.setType(intType)
	mainIncr.setBlock(mainBackEdge)
	mainBackEdge.Instrs = append(mainBackEdge.Instrs, mainIncr)
	spmdAddReferrer(mainIterPhi, mainIncr)
	spmdAddReferrer(laneCountConst, mainIncr)

	// mainCond = mainIncr < alignedBound.
	mainCond := &BinOp{Op: token.LSS}
	mainCond.X = mainIncr
	mainCond.Y = alignedBound
	mainCond.setType(types.Typ[types.Bool])
	mainCond.setBlock(mainBackEdge)
	mainBackEdge.Instrs = append(mainBackEdge.Instrs, mainCond)
	spmdAddReferrer(mainIncr, mainCond)
	spmdAddReferrer(alignedBound, mainCond)

	mainBackIf := &If{Cond: mainCond}
	mainBackIf.setBlock(mainBackEdge)
	mainBackEdge.Instrs = append(mainBackEdge.Instrs, mainBackIf)
	spmdAddReferrer(mainCond, mainBackIf)
	addEdge(mainBackEdge, mainBody)    // true  → loop back to mainBody
	addEdge(mainBackEdge, tailCheck)   // false → proceed to tail check

	// Fill in the back-edge of mainIterPhi now that mainIncr exists.
	// mainBody.Preds = [entry, mainBackEdge], so Edges = [zeroConst, mainIncr].
	mainIterPhi.Edges[1] = mainIncr
	spmdAddReferrer(mainIncr, mainIterPhi)

	// Fill in accumulator back-edges.
	for i, acc := range loop.Accumulators {
		backVal := spmdTranslateValue(acc.BackValue, mainValueMap)
		mainAccPhis[i].Edges[1] = backVal
		spmdAddReferrer(backVal, mainAccPhis[i])
	}

	// --- Populate main early-exit block ---
	// Clone only the non-terminator, non-phi instructions, then wire the
	// original terminator. For a Return block this is just the Return itself;
	// for a Jump→done block it is just the Jump.
	spmdCloneBlock(fn, earlyExitBlock, mainEarlyExit, mainValueMap)
	spmdCloneTerminator(earlyExitBlock, mainEarlyExit, done, mainValueMap)

	// --- Populate tail check ---
	// tailIter = phi [entry: 0, mainBackEdge: mainIncr]
	tailIterPhi := &Phi{Comment: "spmd.tail.iter"}
	tailIterPhi.setType(intType)
	tailIterPhi.Edges = []Value{zeroConst, mainIncr}
	spmdInsertPhiAtFront(tailCheck, tailIterPhi)
	spmdAddReferrer(zeroConst, tailIterPhi)
	spmdAddReferrer(mainIncr, tailIterPhi)

	// Accumulator phis in tailCheck.
	tailAccPhis := make([]*Phi, len(loop.Accumulators))
	tailValueMap := map[Value]Value{loop.IterPhi: tailIterPhi}
	for i, acc := range loop.Accumulators {
		mainAccResult := spmdTranslateValue(acc.BackValue, mainValueMap)
		tailAccPhi := &Phi{Comment: "spmd.tail.acc"}
		tailAccPhi.setType(acc.Phi.Type())
		tailAccPhi.Edges = []Value{acc.InitValue, mainAccResult}
		spmdInsertPhiAtFront(tailCheck, tailAccPhi)
		spmdAddReferrer(acc.InitValue, tailAccPhi)
		spmdAddReferrer(mainAccResult, tailAccPhi)
		tailValueMap[acc.Phi] = tailAccPhi
		tailAccPhis[i] = tailAccPhi
	}

	// hasTail = tailIter < typedBound.
	hasTail := &BinOp{Op: token.LSS}
	hasTail.X = tailIterPhi
	hasTail.Y = typedBound
	hasTail.setType(types.Typ[types.Bool])
	hasTail.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, hasTail)
	spmdAddReferrer(tailIterPhi, hasTail)
	spmdAddReferrer(typedBound, hasTail)

	// Determine tail exit target: need trampoline when accumulators or done phis exist.
	tailExitTarget := done
	var trampoline *BasicBlock
	if len(loop.Accumulators) > 0 || len(done.phis()) > 0 {
		trampoline = fn.newBasicBlock("spmd.trampoline")
		tailExitTarget = trampoline
		loop.TrampolineBlock = trampoline
	}

	tailCheckIf := &If{Cond: hasTail}
	tailCheckIf.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, tailCheckIf)
	spmdAddReferrer(hasTail, tailCheckIf)
	addEdge(tailCheck, tailBody)
	addEdge(tailCheck, tailExitTarget)

	// --- Populate tail body ---
	spmdCloneBlock(fn, body, tailBody, tailValueMap)

	// Wire tail body's cloned uniform-exit If.
	tailBodyCont := spmdFirstChainOrBackEdge(tailChain, tailBackEdge)
	addEdge(tailBody, tailEarlyExit)
	addEdge(tailBody, tailBodyCont)
	tailBodyIf := &If{Cond: spmdTranslateValue(bodyTermIf.Cond, tailValueMap)}
	tailBodyIf.setBlock(tailBody)
	tailBody.Instrs = append(tailBody.Instrs, tailBodyIf)
	spmdAddReferrer(tailBodyIf.Cond, tailBodyIf)

	// --- Populate tail chain blocks ---
	spmdCloneChainBlocks(fn, chainBlocks, tailChain, tailBackEdge, tailValueMap)

	// --- Populate tail back-edge block ---
	spmdCloneBlock(fn, backEdgeBlock, tailBackEdge, tailValueMap)

	// Tail back-edge jumps unconditionally to tailExitTarget (done or trampoline).
	tailBackJump := &Jump{}
	tailBackJump.setBlock(tailBackEdge)
	tailBackEdge.Instrs = append(tailBackEdge.Instrs, tailBackJump)
	addEdge(tailBackEdge, tailExitTarget)

	// --- Populate tail early-exit block ---
	spmdCloneBlock(fn, earlyExitBlock, tailEarlyExit, tailValueMap)
	spmdCloneTerminator(earlyExitBlock, tailEarlyExit, done, tailValueMap)

	// --- Build trampoline when needed ---
	if trampoline != nil {
		accMergeMap := make(map[Value]Value, len(loop.Accumulators)*2)
		for i, acc := range loop.Accumulators {
			tailAccResult := spmdTranslateValue(acc.BackValue, tailValueMap)
			mergePhi := &Phi{Comment: "spmd.acc.merge"}
			mergePhi.setType(acc.Phi.Type())
			// trampoline.Preds = [tailCheck, tailBackEdge] (addEdge order below).
			mergePhi.Edges = []Value{tailAccPhis[i], tailAccResult}
			spmdInsertPhiAtFront(trampoline, mergePhi)
			spmdAddReferrer(tailAccPhis[i], mergePhi)
			spmdAddReferrer(tailAccResult, mergePhi)
			accMergeMap[acc.Phi] = mergePhi
			accMergeMap[acc.BackValue] = mergePhi
			// spmdReplaceAccUses must exclude ALL original blocks that are now dead:
			// body, chain blocks, and backEdgeBlock.
			spmdReplaceAccUsesMulti(acc.Phi, mergePhi, body, chainBlocks, backEdgeBlock)
		}

		trampJump := &Jump{}
		trampJump.setBlock(trampoline)
		trampoline.Instrs = append(trampoline.Instrs, trampJump)
		addEdge(trampoline, done)

		for _, instr := range done.phis() {
			phi := instr.(*Phi)
			// The back-edge value that reached done in the original CFG came
			// from backEdgeBlock, not body (unlike shapeSimple).
			bodyEdge := donePhiBackEdgeEdges[phi]
			if mergeVal, ok2 := accMergeMap[bodyEdge]; ok2 {
				phi.Edges = append(phi.Edges, mergeVal)
				spmdAddReferrer(mergeVal, phi)
				continue
			}
			entryEdge := donePhiEntryEdges[phi]
			if entryEdge == nil {
				entryEdge = bodyEdge
			}
			mainBackEdgeVal := spmdTranslateValue(bodyEdge, mainValueMap)
			tailCheckPhi := &Phi{Comment: "spmd.done.acc"}
			tailCheckPhi.setType(phi.Type())
			tailCheckPhi.Edges = []Value{entryEdge, mainBackEdgeVal}
			spmdInsertPhiAtFront(tailCheck, tailCheckPhi)
			spmdAddReferrer(entryEdge, tailCheckPhi)
			spmdAddReferrer(mainBackEdgeVal, tailCheckPhi)

			tailBodyEdge := spmdTranslateValue(bodyEdge, tailValueMap)
			mergePhi := &Phi{Comment: "spmd.done.merge"}
			mergePhi.setType(phi.Type())
			mergePhi.Edges = []Value{tailCheckPhi, tailBodyEdge}
			spmdInsertPhiAtFront(trampoline, mergePhi)
			spmdAddReferrer(tailCheckPhi, mergePhi)
			spmdAddReferrer(tailBodyEdge, mergePhi)

			phi.Edges = append(phi.Edges, mergePhi)
			spmdAddReferrer(mergePhi, phi)
		}
	}

	// --- Update loop metadata ---
	loop.IsPeeled = true
	loop.MainBodyBlock = mainBody
	loop.TailCheckBlock = tailCheck
	loop.TailBodyBlock = tailBody
	loop.BackEdgeBlock = backEdgeBlock
	loop.AlignedBound = alignedBound
	loop.MainIterPhi = mainIterPhi
	loop.TailIterPhi = tailIterPhi
	loop.BoundValue = typedBound
}

// peelSPMDLoopRangeIndexUniformExit peels a rangeindex loop (go for _, v := range data)
// whose body ends with a uniform (non-varying) If providing an early exit before
// jumping back to the loop block. The rangeindex form keeps its iter phi in a
// dedicated loopBlock (rangeindex.loop), separate from the body block (rangeindex.body).
//
// The concrete SSA shape handled here:
//
//	entry:     ... Jump → loopBlock
//
//	loopBlock: iter = phi [entry: init, body: iter+laneCount]
//	           incr = iter + laneCount    (or iter + 1 × laneCount)
//	           cond = incr < bound
//	           If cond → body, done
//
//	body:      [SPMD ops producing uniformCond ...]
//	           If uniformCond → earlyExit, loopBlock
//
//	earlyExit: Return <val>
//
//	done:      Return <val>
//
// After peeling:
//
//	entry:          alignedBound = bound & ^(lc-1)
//	                If alignedBound > 0 → mainLoopBlock, tailCheck
//
//	mainLoopBlock:  mainIter = phi [entry: 0, mainBody: mainIncr]
//	                [cloned loopBlock non-phi instrs with mainIter substituted]
//	                mainIncr = mainIter + laneCount
//	                mainCond = mainIncr < alignedBound
//	                If mainCond → mainBody, tailCheck
//
//	mainBody:       [clone of body]
//	                If clonedUniformCond → mainEarlyExit, mainLoopBlock
//
//	mainEarlyExit:  Return <translated val>
//
//	tailCheck:      tailIter = phi [entry: 0, mainBody: mainIncr]
//	                hasTail = tailIter < bound
//	                If hasTail → tailBody, done
//
//	tailBody:       [clone of body]
//	                If clonedUniformCond → tailEarlyExit, done
//
//	tailEarlyExit:  Return <translated val>
//
//	done:           Return <val>
func peelSPMDLoopRangeIndexUniformExit(fn *Function, loop *SPMDLoopInfo, laneCount int,
	entry, loopBlock, body, done, earlyExitBlock *BasicBlock) {

	intType := loop.IterPhi.Type()
	typedBound := spmdTypedBound(loop.BoundValue, intType)

	// Extract live bound from the loopBlock's If condition (the incrCond = incr < bound).
	// This mirrors peelSPMDLoopSimple's guard extraction from the entry If.
	// For rangeindex, entry ends with a Jump (not If), so we look at loopBlock's If.
	loopBlockN := len(loopBlock.Instrs)
	if loopBlockN == 0 {
		panic("peelSPMDLoopRangeIndexUniformExit: loopBlock has no instructions")
	}
	loopBlockIf, ok := loopBlock.Instrs[loopBlockN-1].(*If)
	if !ok {
		panic("peelSPMDLoopRangeIndexUniformExit: loopBlock last instruction is not If")
	}
	if guardCond, ok2 := loopBlockIf.Cond.(*BinOp); ok2 && guardCond.Op == token.LSS {
		typedBound = spmdTypedBound(guardCond.Y, intType)
	}

	notMaskConst := NewConst(constant.MakeInt64(int64(^(laneCount-1))), intType)
	laneCountConst := NewConst(constant.MakeInt64(int64(laneCount)), intType)
	zeroConst := NewConst(constant.MakeInt64(0), intType)

	// Create new blocks.
	mainLoopBlock := fn.newBasicBlock("spmd.main.loop")
	mainBody := fn.newBasicBlock("spmd.main.body")
	mainEarlyExit := fn.newBasicBlock("spmd.main.exit")
	tailCheck := fn.newBasicBlock("spmd.tail.check")
	tailBody := fn.newBasicBlock("spmd.tail.body")
	tailEarlyExit := fn.newBasicBlock("spmd.tail.exit")

	// --- Rewrite entry block ---
	// entry ends with a Jump → loopBlock. Remove that Jump and the edge.
	entryN := len(entry.Instrs)
	if entryN == 0 {
		panic("peelSPMDLoopRangeIndexUniformExit: entry block has no instructions")
	}
	entryJump, ok := entry.Instrs[entryN-1].(*Jump)
	if !ok {
		panic("peelSPMDLoopRangeIndexUniformExit: entry block last instruction is not Jump")
	}
	_ = entryJump

	// Remove entry → loopBlock edge.
	loopBlock.removePred(entry)
	entry.Succs = entry.Succs[:0]
	entry.Instrs = entry.Instrs[:entryN-1]

	// Remove loopBlock → done edge (loopBlock's If had Succs=[body,done]).
	// Save loopBlock→done phi edges before disconnecting.
	donePhiLoopEdges := map[*Phi]Value{}
	for _, instr := range done.phis() {
		phi := instr.(*Phi)
		for j, pred := range done.Preds {
			if pred == loopBlock {
				donePhiLoopEdges[phi] = phi.Edges[j]
			}
		}
	}
	done.removePred(loopBlock)

	// Remove body → loopBlock edge. Body's If had Succs=[earlyExit, loopBlock].
	loopBlock.removePred(body)

	// Remove earlyExitBlock from done.Preds if it was a direct Jump→done.
	if len(earlyExitBlock.Succs) == 1 && earlyExitBlock.Succs[0] == done {
		done.removePred(earlyExitBlock)
	}

	// Build aligned-bound computation in entry.
	alignedBound := &BinOp{Op: token.AND}
	alignedBound.X = typedBound
	alignedBound.Y = notMaskConst
	alignedBound.setType(intType)
	alignedBound.setBlock(entry)
	entry.Instrs = append(entry.Instrs, alignedBound)
	spmdAddReferrer(typedBound, alignedBound)
	spmdAddReferrer(notMaskConst, alignedBound)

	hasMain := &BinOp{Op: token.GTR}
	hasMain.X = alignedBound
	hasMain.Y = zeroConst
	hasMain.setType(types.Typ[types.Bool])
	hasMain.setBlock(entry)
	entry.Instrs = append(entry.Instrs, hasMain)
	spmdAddReferrer(alignedBound, hasMain)
	spmdAddReferrer(zeroConst, hasMain)

	entryIf := &If{Cond: hasMain}
	entryIf.setBlock(entry)
	entry.Instrs = append(entry.Instrs, entryIf)
	spmdAddReferrer(hasMain, entryIf)
	addEdge(entry, mainLoopBlock)
	addEdge(entry, tailCheck)

	// --- Populate mainLoopBlock ---
	// Build mainIterPhi: phi [entry: 0, mainBody: mainIncr]
	// Edges[1] (mainIncr) is filled after it's created.
	mainIterPhi := &Phi{Comment: "spmd.main.iter"}
	mainIterPhi.setType(intType)
	mainIterPhi.Edges = []Value{zeroConst, nil}
	spmdInsertPhiAtFront(mainLoopBlock, mainIterPhi)
	spmdAddReferrer(zeroConst, mainIterPhi)

	// Accumulator phis in mainLoopBlock.
	mainAccPhis := make([]*Phi, len(loop.Accumulators))
	mainLoopValueMap := map[Value]Value{loop.IterPhi: mainIterPhi}
	for i, acc := range loop.Accumulators {
		mainAccPhi := &Phi{Comment: "spmd.main.acc"}
		mainAccPhi.setType(acc.Phi.Type())
		mainAccPhi.Edges = []Value{acc.InitValue, nil}
		spmdInsertPhiAtFront(mainLoopBlock, mainAccPhi)
		spmdAddReferrer(acc.InitValue, mainAccPhi)
		mainLoopValueMap[acc.Phi] = mainAccPhi
		mainAccPhis[i] = mainAccPhi
	}

	// Build mainIncr = mainIterPhi + laneCount.
	// mainLoopBlock is constructed entirely from scratch: phi + mainIncr + mainCond + If.
	// We do NOT clone anything from loopBlock — loopBlock may contain ChangeType/Convert/UnOp
	// instructions between the original IncrBinOp and the bound check, and cloning them
	// would produce references to values in the now-unreachable loopBlock.  Instead,
	// mainCond compares mainIncr directly against alignedBound (computed in entry), so no
	// loopBlock instructions are needed here.
	mainIncr := &BinOp{Op: token.ADD}
	mainIncr.X = mainIterPhi
	mainIncr.Y = laneCountConst
	mainIncr.setType(intType)
	mainIncr.setBlock(mainLoopBlock)
	mainLoopBlock.Instrs = append(mainLoopBlock.Instrs, mainIncr)
	spmdAddReferrer(mainIterPhi, mainIncr)
	spmdAddReferrer(laneCountConst, mainIncr)

	// mainCond = mainIncr < alignedBound.
	// alignedBound is always the same integer type as iter, so no conversion needed.
	mainCond := &BinOp{Op: token.LSS}
	mainCond.X = mainIncr
	mainCond.Y = alignedBound
	mainCond.setType(types.Typ[types.Bool])
	mainCond.setBlock(mainLoopBlock)
	mainLoopBlock.Instrs = append(mainLoopBlock.Instrs, mainCond)
	spmdAddReferrer(mainIncr, mainCond)
	spmdAddReferrer(alignedBound, mainCond)

	// If mainCond → mainBody, tailCheck.
	mainLoopIf := &If{Cond: mainCond}
	mainLoopIf.setBlock(mainLoopBlock)
	mainLoopBlock.Instrs = append(mainLoopBlock.Instrs, mainLoopIf)
	spmdAddReferrer(mainCond, mainLoopIf)
	addEdge(mainLoopBlock, mainBody)
	addEdge(mainLoopBlock, tailCheck)

	// Fill in the back-edge of mainIterPhi now that mainBody is the predecessor.
	// mainLoopBlock.Preds = [entry, mainBody] (addEdge order will be confirmed below).
	// We fill mainIterPhi.Edges[1] = mainIncr AFTER wiring mainBody → mainLoopBlock.

	// --- Populate mainBody ---
	// The body block (rangeindex.body) accesses slice elements via IncrBinOp
	// (the "incr = iter + laneCount" value from loopBlock). Since mainIterPhi
	// starts at 0 and IncrBinOp in the original starts at 0 (first body visit
	// with iter=-laneCount → incr=0), we map IncrBinOp → mainIterPhi so that
	// body references to incrBinOp correctly use the current group's base index.
	mainValueMap := map[Value]Value{
		loop.IterPhi:   mainIterPhi,
		loop.IncrBinOp: mainIterPhi,
	}
	// Copy all accumulator phi mappings from loopBlock map to body map.
	for i, acc := range loop.Accumulators {
		mainValueMap[acc.Phi] = mainAccPhis[i]
	}

	// Clone body instructions into mainBody.
	spmdCloneBlock(fn, body, mainBody, mainValueMap)

	// Wire mainBody's cloned uniform-exit If:
	//   true  branch → mainEarlyExit
	//   false branch → mainLoopBlock (continue loop)
	bodyTermIf := body.Instrs[len(body.Instrs)-1].(*If)
	mainBodyIf := &If{Cond: spmdTranslateValue(bodyTermIf.Cond, mainValueMap)}
	mainBodyIf.setBlock(mainBody)
	mainBody.Instrs = append(mainBody.Instrs, mainBodyIf)
	spmdAddReferrer(mainBodyIf.Cond, mainBodyIf)
	addEdge(mainBody, mainEarlyExit)
	addEdge(mainBody, mainLoopBlock)

	// Now wire mainLoopBlock ← mainBody back-edge:
	// mainLoopBlock.Preds = [entry, mainBody] in addEdge order.
	mainIterPhi.Edges[1] = mainIncr
	spmdAddReferrer(mainIncr, mainIterPhi)

	// Fill accumulator back-edges.
	for i, acc := range loop.Accumulators {
		backVal := spmdTranslateValue(acc.BackValue, mainValueMap)
		mainAccPhis[i].Edges[1] = backVal
		spmdAddReferrer(backVal, mainAccPhis[i])
	}

	// --- Populate mainEarlyExit ---
	spmdCloneBlock(fn, earlyExitBlock, mainEarlyExit, mainValueMap)
	spmdCloneTerminator(earlyExitBlock, mainEarlyExit, done, mainValueMap)

	// --- Populate tailCheck ---
	// tailIter = phi [entry: 0, mainLoopBlock: mainIterPhi]
	// tailCheck.Preds = [entry, mainLoopBlock].
	// When entry → tailCheck (no main loop, alignedBound == 0), tail starts at 0.
	// When mainLoopBlock → tailCheck (main loop finished), tail starts at mainIterPhi
	// which holds the first unprocessed group index (e.g., 1020 when bound=1024 and
	// laneCount=4). Using mainIncr (= mainIterPhi + laneCount) would overshoot: it
	// equals alignedBound at loop exit, so hasTail = alignedBound < bound is false
	// and the tail is skipped even when bound is not a multiple of laneCount.
	tailIterPhi := &Phi{Comment: "spmd.tail.iter"}
	tailIterPhi.setType(intType)
	tailIterPhi.Edges = []Value{zeroConst, mainIterPhi}
	spmdInsertPhiAtFront(tailCheck, tailIterPhi)
	spmdAddReferrer(zeroConst, tailIterPhi)
	spmdAddReferrer(mainIterPhi, tailIterPhi)

	// Accumulator phis in tailCheck.
	tailAccPhis := make([]*Phi, len(loop.Accumulators))
	// Map both IterPhi and IncrBinOp to tailIterPhi so body instructions that
	// reference the original IncrBinOp (rangeindex's "incr = iter + laneCount"
	// in loopBlock) use the peeled iter value instead of the unreachable original.
	tailValueMap := map[Value]Value{
		loop.IterPhi:   tailIterPhi,
		loop.IncrBinOp: tailIterPhi,
	}
	for i, acc := range loop.Accumulators {
		mainAccResult := spmdTranslateValue(acc.BackValue, mainValueMap)
		tailAccPhi := &Phi{Comment: "spmd.tail.acc"}
		tailAccPhi.setType(acc.Phi.Type())
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

	// Determine tail exit target (trampoline needed when accumulators or done phis).
	tailExitTarget := done
	var trampoline *BasicBlock
	if len(loop.Accumulators) > 0 || len(done.phis()) > 0 {
		trampoline = fn.newBasicBlock("spmd.trampoline")
		tailExitTarget = trampoline
		loop.TrampolineBlock = trampoline
	}

	hasTailIf := &If{Cond: hasTail}
	hasTailIf.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, hasTailIf)
	spmdAddReferrer(hasTail, hasTailIf)
	addEdge(tailCheck, tailBody)
	addEdge(tailCheck, tailExitTarget)

	// --- Populate tailBody ---
	spmdCloneBlock(fn, body, tailBody, tailValueMap)

	// Wire tailBody's cloned uniform-exit If:
	//   true  branch → tailEarlyExit
	//   false branch → tailExitTarget (done or trampoline, not back to loop)
	tailBodyIf := &If{Cond: spmdTranslateValue(bodyTermIf.Cond, tailValueMap)}
	tailBodyIf.setBlock(tailBody)
	tailBody.Instrs = append(tailBody.Instrs, tailBodyIf)
	spmdAddReferrer(tailBodyIf.Cond, tailBodyIf)
	addEdge(tailBody, tailEarlyExit)
	addEdge(tailBody, tailExitTarget)

	// --- Populate tailEarlyExit ---
	spmdCloneBlock(fn, earlyExitBlock, tailEarlyExit, tailValueMap)
	spmdCloneTerminator(earlyExitBlock, tailEarlyExit, done, tailValueMap)

	// --- Build trampoline when needed ---
	if trampoline != nil {
		accMergeMap := make(map[Value]Value, len(loop.Accumulators)*2)
		for i, acc := range loop.Accumulators {
			tailAccResult := spmdTranslateValue(acc.BackValue, tailValueMap)
			mergePhi := &Phi{Comment: "spmd.acc.merge"}
			mergePhi.setType(acc.Phi.Type())
			// trampoline.Preds = [tailCheck, tailBody] (addEdge order).
			mergePhi.Edges = []Value{tailAccPhis[i], tailAccResult}
			spmdInsertPhiAtFront(trampoline, mergePhi)
			spmdAddReferrer(tailAccPhis[i], mergePhi)
			spmdAddReferrer(tailAccResult, mergePhi)
			accMergeMap[acc.Phi] = mergePhi
			accMergeMap[acc.BackValue] = mergePhi
			// Dead region is {body, earlyExitBlock, loopBlock} — all become
			// unreachable after peeling. Exclude all three so stale referrers
			// aren't recorded on mergePhi.
			spmdReplaceAccUsesMulti(acc.Phi, mergePhi, body, []*BasicBlock{earlyExitBlock}, loopBlock)
		}

		trampJump := &Jump{}
		trampJump.setBlock(trampoline)
		trampoline.Instrs = append(trampoline.Instrs, trampJump)
		addEdge(trampoline, done)

		// Wire done-block phis for the new trampoline predecessor.
		// In rangeindex, the original done-predecessor was loopBlock (not body).
		for _, instr := range done.phis() {
			phi := instr.(*Phi)
			loopEdge := donePhiLoopEdges[phi]
			if mergeVal, ok3 := accMergeMap[loopEdge]; ok3 {
				phi.Edges = append(phi.Edges, mergeVal)
				spmdAddReferrer(mergeVal, phi)
				continue
			}
			// Non-accumulator done-block phi: build tailCheck + trampoline merge phis.
			entryEdge := donePhiLoopEdges[phi] // rangeindex: entry path through loopBlock
			if entryEdge == nil {
				entryEdge = loopEdge
			}
			mainLoopEdgeVal := spmdTranslateValue(loopEdge, mainLoopValueMap)
			tailCheckPhi := &Phi{Comment: "spmd.done.acc"}
			tailCheckPhi.setType(phi.Type())
			tailCheckPhi.Edges = []Value{entryEdge, mainLoopEdgeVal}
			spmdInsertPhiAtFront(tailCheck, tailCheckPhi)
			spmdAddReferrer(entryEdge, tailCheckPhi)
			spmdAddReferrer(mainLoopEdgeVal, tailCheckPhi)

			tailBodyEdge := spmdTranslateValue(loopEdge, tailValueMap)
			mergePhi := &Phi{Comment: "spmd.done.merge"}
			mergePhi.setType(phi.Type())
			mergePhi.Edges = []Value{tailCheckPhi, tailBodyEdge}
			spmdInsertPhiAtFront(trampoline, mergePhi)
			spmdAddReferrer(tailCheckPhi, mergePhi)
			spmdAddReferrer(tailBodyEdge, mergePhi)

			phi.Edges = append(phi.Edges, mergePhi)
			spmdAddReferrer(mergePhi, phi)
		}
	}

	// --- Update loop metadata ---
	loop.IsPeeled = true
	loop.MainBodyBlock = mainBody
	loop.TailCheckBlock = tailCheck
	loop.TailBodyBlock = tailBody
	loop.BackEdgeBlock = mainLoopBlock
	loop.AlignedBound = alignedBound
	loop.MainIterPhi = mainIterPhi
	loop.TailIterPhi = tailIterPhi
	loop.BoundValue = typedBound
}

// spmdFirstChainOrBackEdge returns the first clone chain block if any, or the
// back-edge block when the chain is empty (body connects directly to back-edge).
func spmdFirstChainOrBackEdge(chain []*BasicBlock, backEdge *BasicBlock) *BasicBlock {
	if len(chain) > 0 {
		return chain[0]
	}
	return backEdge
}

// spmdCloneChainBlocks clones a sequence of straight-line chain blocks into
// their corresponding target blocks, wiring each clone to the next. The last
// clone's Jump points to finalDest (the cloned back-edge block).
func spmdCloneChainBlocks(fn *Function, origChain, cloneChain []*BasicBlock,
	finalDest *BasicBlock, valueMap map[Value]Value) {
	for i, orig := range origChain {
		dst := cloneChain[i]
		spmdCloneBlock(fn, orig, dst, valueMap)

		// Determine the successor of this clone: next chain block or finalDest.
		var succ *BasicBlock
		if i+1 < len(cloneChain) {
			succ = cloneChain[i+1]
		} else {
			succ = finalDest
		}
		// Wire: emit a Jump to succ and add the CFG edge.
		j := &Jump{}
		j.setBlock(dst)
		dst.Instrs = append(dst.Instrs, j)
		addEdge(dst, succ)
	}
}

// spmdCloneTerminator copies the terminator of srcBlock into dstBlock,
// translating operand values through valueMap. For Return terminators, result
// values are translated. For Jump-to-done terminators, the Jump targets done
// directly without translation (done is the loop exit; it is already correct).
// Panic terminators are not expected here and are left as-is.
func spmdCloneTerminator(srcBlock, dstBlock, done *BasicBlock, valueMap map[Value]Value) {
	if len(srcBlock.Instrs) == 0 {
		return
	}
	switch term := srcBlock.Instrs[len(srcBlock.Instrs)-1].(type) {
	case *Return:
		newRet := &Return{}
		newRet.Results = make([]Value, len(term.Results))
		for i, r := range term.Results {
			newRet.Results[i] = spmdTranslateValue(r, valueMap)
			spmdAddReferrer(newRet.Results[i], newRet)
		}
		newRet.pos = term.pos
		newRet.setBlock(dstBlock)
		dstBlock.Instrs = append(dstBlock.Instrs, newRet)

	case *Jump:
		// A Jump in an exit block goes to done; keep targeting done.
		j := &Jump{}
		j.setBlock(dstBlock)
		dstBlock.Instrs = append(dstBlock.Instrs, j)
		addEdge(dstBlock, done)
	}
}

// spmdReplaceAccUsesMulti is like spmdReplaceAccUses but excludes instructions
// in any of the provided original loop body blocks (all about to become
// unreachable after peeling). This is needed for shapeUniformExit where the
// dead region spans body + chain blocks + backEdgeBlock.
func spmdReplaceAccUsesMulti(oldPhi *Phi, newVal Value, body *BasicBlock, chainBlocks []*BasicBlock, backEdge *BasicBlock) {
	// Build a set of excluded blocks for O(1) lookup.
	excluded := map[*BasicBlock]bool{body: true, backEdge: true}
	for _, cb := range chainBlocks {
		excluded[cb] = true
	}

	pxrefs := oldPhi.Referrers()
	if pxrefs == nil {
		return
	}
	pyrefs := newVal.Referrers()
	var remaining []Instruction
	for _, instr := range *pxrefs {
		if excluded[instr.Block()] {
			remaining = append(remaining, instr)
			continue
		}
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
			*SPMDExtractMask, *IndexAddr, *FieldAddr, *Convert, *ChangeType, *Call, *Alloc,
			*MakeInterface, *SPMDVectorFromMemory:
			// Handled by spmdCloneBlock; OK.
		default:
			return false
		}
	}
	return true
}

// loopBodyShape classifies the structure of an SPMD loop body.
type loopBodyShape int

const (
	// shapeSimple: body block has exactly two successors, itself and done.
	// This is the merged rangeint form; the existing fast path handles it.
	shapeSimple loopBodyShape = iota

	// shapeUniformExit: body ends with a uniform (non-varying) If; one
	// successor is an exit block (Return or Jump to done), the other leads
	// through zero or more cloneable chain blocks to a back-edge block whose
	// If has Succs[0]==body and Succs[1]==done.
	shapeUniformExit

	// shapeUnpeelable: any other shape; peeling is skipped.
	shapeUnpeelable
)

// spmdClassifyLoopBody inspects the successors of body and returns the loop
// body shape. For shapeUniformExit it also returns:
//   - exitBlock: the block reached when the uniform condition is true
//     (either a Return block or a Jump-to-done block).
//   - chainBlocks: zero or more intermediate blocks between the body's
//     continuation successor and the back-edge block, in traversal order.
//   - backEdge: the block whose If has Succs[0]==body and Succs[1]==done.
//
// All chainBlocks (and body itself) must pass spmdBodyIsCloneable.
// exitBlock must either terminate with Return or contain a single Jump to done.
func spmdClassifyLoopBody(body, done *BasicBlock) (shape loopBodyShape, exitBlock *BasicBlock, chainBlocks []*BasicBlock, backEdge *BasicBlock) {
	// shapeSimple: body is its own back-edge.
	if len(body.Succs) == 2 && body.Succs[0] == body && body.Succs[1] == done {
		return shapeSimple, nil, nil, nil
	}

	// shapeUniformExit: body must end with a non-varying If with exactly two successors.
	if len(body.Succs) != 2 {
		return shapeUnpeelable, nil, nil, nil
	}
	bodyIf, ok := body.Instrs[len(body.Instrs)-1].(*If)
	if !ok || bodyIf.IsVarying {
		// Either no If terminator or a varying If — not a uniform exit.
		return shapeUnpeelable, nil, nil, nil
	}

	// Determine which successor is the exit and which is the continuation.
	// exitBlock must be a dedicated early-exit block: Return-terminated, or a
	// single-Jump block to done. The done block itself is not a valid exitBlock
	// because removing it from done.Preds would corrupt the loop structure.
	// contStart is where the chain to the back-edge begins.
	var contStart *BasicBlock
	for _, succ := range [2]*BasicBlock{body.Succs[0], body.Succs[1]} {
		other := body.Succs[0]
		if succ == body.Succs[0] {
			other = body.Succs[1]
		}
		// succ == done means the body's If exits directly to done — that is
		// a direct break/continue, not an early-exit block that can be cloned.
		if succ == done {
			continue
		}
		if spmdIsExitBlock(succ, done) {
			exitBlock = succ
			contStart = other
			break
		}
	}
	if exitBlock == nil {
		return shapeUnpeelable, nil, nil, nil
	}

	// Walk from contStart, collecting chain blocks, until we find the
	// back-edge block (one whose If has Succs[0]==body and Succs[1]==done).
	// Each intermediate block must have exactly one successor (be straight-line),
	// and every block visited must pass spmdBodyIsCloneable.
	//
	// Limit the walk to avoid infinite loops in malformed CFGs; in practice
	// the chain is at most a handful of blocks.
	const maxChainLen = 16
	cur := contStart
	for i := 0; i < maxChainLen; i++ {
		if !spmdBodyIsCloneable(cur) {
			return shapeUnpeelable, nil, nil, nil
		}
		// Check if cur is the back-edge block.
		if len(cur.Succs) == 2 && cur.Succs[0] == body && cur.Succs[1] == done {
			backEdge = cur
			return shapeUniformExit, exitBlock, chainBlocks, backEdge
		}
		// Must be a straight-line block to continue the chain.
		if len(cur.Succs) != 1 {
			return shapeUnpeelable, nil, nil, nil
		}
		chainBlocks = append(chainBlocks, cur)
		cur = cur.Succs[0]
	}
	return shapeUnpeelable, nil, nil, nil
}

// spmdIsExitBlock reports whether block is an exit from the loop body:
// either it terminates with a Return, or it is a single-instruction Jump to done.
func spmdIsExitBlock(block, done *BasicBlock) bool {
	if len(block.Instrs) == 0 {
		return false
	}
	switch block.Instrs[len(block.Instrs)-1].(type) {
	case *Return:
		return true
	case *Jump:
		return len(block.Succs) == 1 && block.Succs[0] == done
	}
	return false
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
