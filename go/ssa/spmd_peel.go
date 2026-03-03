// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

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
