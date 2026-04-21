// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/ast"
	"go/token"
	"go/types"
)

// spmdFieldAddrResultType returns the SSA-level type for a FieldAddr whose
// base has type baseT and whose struct field has plain type fieldT.
//
//   - *Varying[S] base  → *Varying[fieldT]  (existing behavior; uniform ptr to varying struct).
//   - Varying[*S] base  → Varying[*fieldT]  (new; per-lane pointer vector).
//   - Otherwise         → *fieldT           (normal Go field-address).
//
// This mirrors the asymmetry between *Varying[T] and Varying[*T] at the
// type-checker level (see go/types/call_ext_spmd.go spmdWrapFieldType).
func spmdFieldAddrResultType(baseT, fieldT types.Type) types.Type {
	// *Varying[S] base → *Varying[fieldT]
	if ptr, ok := baseT.(*types.Pointer); ok {
		if _, ok := ptr.Elem().(*types.SPMDType); ok {
			return types.NewPointer(types.NewVarying(fieldT))
		}
	}
	// Varying[*S] base → Varying[*fieldT]
	if sv, ok := baseT.(*types.SPMDType); ok {
		if _, ok := sv.Elem().(*types.Pointer); ok {
			return types.NewVarying(types.NewPointer(fieldT))
		}
	}
	// Normal Go.
	return types.NewPointer(fieldT)
}

// exprHasSPMDType reports whether the AST expression e has or involves
// *types.SPMDType. It checks the type-checker type first, then recurses
// into sub-expressions for cases where conversions strip SPMDType
// (e.g., byte(varyingExpr) has type byte, not Varying[byte]).
func exprHasSPMDType(fn *Function, e ast.Expr) bool {
	if fn.info == nil {
		return false
	}
	if t := fn.info.TypeOf(e); t != nil {
		if _, ok := t.(*types.SPMDType); ok {
			return true
		}
	}
	switch e := e.(type) {
	case *ast.BinaryExpr:
		return exprHasSPMDType(fn, e.X) || exprHasSPMDType(fn, e.Y)
	case *ast.UnaryExpr:
		return exprHasSPMDType(fn, e.X)
	case *ast.ParenExpr:
		return exprHasSPMDType(fn, e.X)
	case *ast.CallExpr:
		// Only recurse into type conversions (byte(varyingExpr)),
		// not function calls (uniformFn(varyingArg)) where the return
		// type determines varying-ness, already checked by TypeOf above.
		if tv, ok := fn.info.Types[e.Fun]; ok && tv.IsType() {
			for _, arg := range e.Args {
				if exprHasSPMDType(fn, arg) {
					return true
				}
			}
		}
	case *ast.IndexExpr:
		return exprHasSPMDType(fn, e.X) || exprHasSPMDType(fn, e.Index)
	}
	return false
}

// isSideEffectFreeBoolExpr reports whether the boolean expression e can be
// evaluated unconditionally without side effects. Used to flatten varying
// &&/|| to bitwise AND/OR instead of short-circuit branches.
//
// Safe expressions: comparisons (with safe operands), boolean literals,
// variable references, field selectors, unary !, parenthesized expressions,
// and nested &&/||.
// Unsafe: function calls, index expressions, channel ops, etc.
func isSideEffectFreeBoolExpr(fn *Function, e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.BinaryExpr:
		switch e.Op {
		case token.LAND, token.LOR:
			return isSideEffectFreeBoolExpr(fn, e.X) && isSideEffectFreeBoolExpr(fn, e.Y)
		case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
			// Comparisons are side-effect-free only if their operands are too.
			return isSideEffectFreeExpr(e.X) && isSideEffectFreeExpr(e.Y)
		default:
			return false
		}
	case *ast.UnaryExpr:
		if e.Op == token.NOT {
			return isSideEffectFreeBoolExpr(fn, e.X)
		}
		return false
	case *ast.ParenExpr:
		return isSideEffectFreeBoolExpr(fn, e.X)
	case *ast.Ident:
		return true
	case *ast.BasicLit:
		return true
	default:
		return false
	}
}

// isSideEffectFreeExpr reports whether a general expression (not necessarily
// boolean) can be evaluated without side effects. Used to check comparison
// operands in isSideEffectFreeBoolExpr.
func isSideEffectFreeExpr(e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.Ident:
		return true
	case *ast.BasicLit:
		return true
	case *ast.SelectorExpr:
		return isSideEffectFreeExpr(e.X)
	case *ast.ParenExpr:
		return isSideEffectFreeExpr(e.X)
	case *ast.UnaryExpr:
		return isSideEffectFreeExpr(e.X)
	case *ast.BinaryExpr:
		// Arithmetic ops (+, -, *, etc.) are side-effect-free.
		return isSideEffectFreeExpr(e.X) && isSideEffectFreeExpr(e.Y)
	case *ast.CallExpr:
		// Function calls may have side effects.
		return false
	case *ast.IndexExpr:
		// Index expressions can panic on out-of-bounds.
		return false
	default:
		return false
	}
}

// resolveSPMDSwitchChains updates SPMDSwitchChain block pointers after
// optimizeBlocks has potentially fused or eliminated blocks. Called from
// finishBody right after optimizeBlocks and removeNilBlocks.
func resolveSPMDSwitchChains(fn *Function) {
	for _, chain := range fn.SPMDSwitchChains {
		chain.DoneBlock = resolveBlock(fn, chain.DoneBlock)
		if chain.DefaultBlock != nil {
			chain.DefaultBlock = resolveBlock(fn, chain.DefaultBlock)
		}
	}
}

// resolveSPMDBooleanChains updates SPMDBooleanChain block pointers after
// optimizeBlocks has potentially fused or eliminated blocks.
func resolveSPMDBooleanChains(fn *Function) {
	for _, chain := range fn.SPMDBooleanChains {
		chain.ThenBlock = resolveBlock(fn, chain.ThenBlock)
		chain.ElseBlock = resolveBlock(fn, chain.ElseBlock)
	}
}

// resolveBlock returns the surviving block after optimizeBlocks may have
// eliminated the original block via fuseBlocks or jumpThreading.
// For fused blocks, the instructions' Block() returns the absorbing block.
// For jump-threaded blocks, the successor is the redirect target.
func resolveBlock(fn *Function, block *BasicBlock) *BasicBlock {
	if blockInFunction(fn, block) {
		return block
	}
	// fuseBlocks: instructions moved to absorbing block via setBlock.
	if len(block.Instrs) > 0 {
		if surviving := block.Instrs[0].Block(); surviving != block && blockInFunction(fn, surviving) {
			return surviving
		}
	}
	// jumpThreading: successor is the redirect target.
	if len(block.Succs) > 0 {
		return resolveBlock(fn, block.Succs[0])
	}
	return block
}
