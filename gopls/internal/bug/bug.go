// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bug provides utilities for reporting internal bugs, and being
// notified when they occur.
//
// Philosophically, because gopls runs as a sidecar process that the user does
// not directly control, sometimes it keeps going on broken invariants rather
// than panicking. In those cases, bug reports provide a mechanism to alert
// developers and capture relevant metadata.
package bug

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
)

// PanicOnBugs controls whether to panic when bugs are reported.
//
// It may be set to true during testing.
var PanicOnBugs = false

var (
	mu        sync.Mutex
	exemplars map[string]Bug
	waiters   []chan<- Bug
)

// A Bug represents an unexpected event or broken invariant. They are used for
// capturing metadata that helps us understand the event.
type Bug struct {
	File        string // file containing the call to bug.Report
	Line        int    // line containing the call to bug.Report
	Description string // description of the bug
	Key         string // key identifying the bug (file:line if available)
	Stack       string // call stack
}

// Reportf reports a formatted bug message.
func Reportf(format string, args ...interface{}) {
	report(fmt.Sprintf(format, args...))
}

// Errorf calls fmt.Errorf for the given arguments, and reports the resulting
// error message as a bug.
func Errorf(format string, args ...interface{}) error {
	err := fmt.Errorf(format, args...)
	report(err.Error())
	return err
}

// Report records a new bug encountered on the server.
// It uses reflection to report the position of the immediate caller.
func Report(description string) {
	report(description)
}

func report(description string) {
	_, file, line, ok := runtime.Caller(2) // all exported reporting functions call report directly

	key := "<missing callsite>"
	if ok {
		key = fmt.Sprintf("%s:%d", file, line)
	}

	if PanicOnBugs {
		panic(fmt.Sprintf("%s: %s", key, description))
	}

	bug := Bug{
		File:        file,
		Line:        line,
		Description: description,
		Key:         key,
		Stack:       string(debug.Stack()),
	}

	mu.Lock()
	defer mu.Unlock()

	if exemplars == nil {
		exemplars = make(map[string]Bug)
	}

	if _, ok := exemplars[key]; !ok {
		exemplars[key] = bug // capture one exemplar per key
	}

	for _, waiter := range waiters {
		waiter <- bug
	}
	waiters = nil
}

// Notify returns a channel that will be sent the next bug to occur on the
// server. This channel only ever receives one bug.
func Notify() <-chan Bug {
	mu.Lock()
	defer mu.Unlock()

	ch := make(chan Bug, 1) // 1-buffered so that bug reporting is non-blocking
	waiters = append(waiters, ch)
	return ch
}

// List returns a slice of bug exemplars -- the first bugs to occur at each
// callsite.
func List() []Bug {
	mu.Lock()
	defer mu.Unlock()

	var bugs []Bug

	for _, bug := range exemplars {
		bugs = append(bugs, bug)
	}

	sort.Slice(bugs, func(i, j int) bool {
		return bugs[i].Key < bugs[j].Key
	})

	return bugs
}
