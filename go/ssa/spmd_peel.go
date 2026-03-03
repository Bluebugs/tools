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

		case *IndexAddr:
			newInstr := &IndexAddr{}
			newInstr.X = spmdTranslateValue(v.X, valueMap)
			newInstr.Index = spmdTranslateValue(v.Index, valueMap)
			newInstr.setType(v.Type())
			newInstr.setBlock(dstBlock)
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

	// Clone body instructions, mapping the original IterPhi to mainIterPhi.
	mainValueMap := map[Value]Value{loop.IterPhi: mainIterPhi}
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

	// hasTail = tailIterPhi < typedBound.
	hasTail := &BinOp{Op: token.LSS}
	hasTail.X = tailIterPhi
	hasTail.Y = typedBound
	hasTail.setType(types.Typ[types.Bool])
	hasTail.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, hasTail)
	spmdAddReferrer(tailIterPhi, hasTail)
	spmdAddReferrer(typedBound, hasTail)

	// If(hasTail) → tailBody, done.
	tailCheckIf := &If{Cond: hasTail}
	tailCheckIf.setBlock(tailCheck)
	tailCheck.Instrs = append(tailCheck.Instrs, tailCheckIf)
	spmdAddReferrer(hasTail, tailCheckIf)

	// Wire tailCheck → tailBody and tailCheck → done.
	addEdge(tailCheck, tailBody)
	addEdge(tailCheck, done)

	// --- Populate tail body ---
	tailValueMap := map[Value]Value{loop.IterPhi: tailIterPhi}
	spmdCloneBlock(fn, body, tailBody, tailValueMap)

	// Jump → done.
	tailJump := &Jump{}
	tailJump.setBlock(tailBody)
	tailBody.Instrs = append(tailBody.Instrs, tailJump)

	// Wire tailBody → done.
	addEdge(tailBody, done)

	// --- Update loop metadata ---
	loop.IsPeeled = true
	loop.MainBodyBlock = mainBody
	loop.TailCheckBlock = tailCheck
	loop.TailBodyBlock = tailBody
	loop.AlignedBound = alignedBound
	loop.TailIterPhi = tailIterPhi
	// TrampolineBlock remains nil for loops without accumulators.
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
