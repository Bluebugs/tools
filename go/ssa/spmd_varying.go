// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/ast"
	"go/types"
)

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
