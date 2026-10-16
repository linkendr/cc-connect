//go:build unix

package opencode

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

// startFakeTurn starts a session whose fake CLI prints a JSON text event
// carrying a background child's PID (readiness signal), then keeps running.
func startFakeTurn(t *testing.T, script string) (*opencodeSession, int) {
	t.Helper()
	s, err := newOpencodeSession(context.Background(), writeFakeOpencode(t, script), nil, t.TempDir(), "", "default", "", "", nil)
	if err != nil {
		t.Fatalf("newOpencodeSession: %v", err)
	}
	if err := s.Send("hi", "m1", nil, nil); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case evt := <-s.Events():
		pid, err := strconv.Atoi(strings.TrimSpace(evt.Content))
		if evt.Type != core.EventText || err != nil {
			t.Fatalf("readiness event = %+v", evt)
		}
		return s, pid
	case <-time.After(5 * time.Second):
		t.Fatal("no readiness event")
		return nil, 0
	}
}

func alive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}

// Close kills the whole process group: a grandchild holding the pipe dies,
// the read loop exits, and Close returns promptly without error.
func TestOpencodeCloseKillsProcessGroup(t *testing.T) {
	s, pid := startFakeTurn(t, "sleep 30 &\n"+`echo "{\"type\":\"text\",\"part\":{\"text\":\"$!\"}}"`+"\nsleep 30")
	start := time.Now()
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	t.Logf("Close took %v", time.Since(start))
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("Close took %v", d)
	}
	deadline := time.Now().Add(2 * time.Second)
	for alive(pid) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	t.Logf("grandchild gone after %v", time.Since(start))
	if alive(pid) {
		_ = syscall.Kill(pid, syscall.SIGKILL)
		t.Fatal("grandchild survived Close")
	}
}

// (a) A descendant that left the process group and redirected stderr keeps
// stdout open: Close closes the session's read ends, so the read loop and
// the stderr reader exit and Close returns within its bound.
func TestOpencodeCloseUnblocksEscapedStdoutHolder(t *testing.T) {
	if _, err := exec.LookPath("perl"); err != nil {
		t.Skip("perl not available")
	}
	s, pid := startFakeTurn(t, `perl -e '$|=1; setpgrp(0,0); open(STDERR, ">/dev/null"); print "{\"type\":\"text\",\"part\":{\"text\":\"$$\"}}\n"; sleep 30' &`+"\nsleep 30")
	t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
	start := time.Now()
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if d := time.Since(start); d > closeWaitTimeout {
		t.Errorf("Close took %v", d)
	}
	exited := make(chan struct{})
	go func() { s.wg.Wait(); close(exited) }()
	select {
	case <-exited:
	case <-time.After(time.Second):
		t.Fatal("reader goroutines still running after Close")
	}
}

// (b) The CLI already exited 0, leaving a same-group descendant that holds
// no pipe: Close still kills the group.
func TestOpencodeCloseKillsLeftoverGroupAfterExit(t *testing.T) {
	script := `sleep 30 </dev/null >/dev/null 2>&1 &` + "\n" +
		`echo "{\"type\":\"text\",\"part\":{\"text\":\"$!\"}}"` + "\nexit 0"
	s, pid := startFakeTurn(t, script)
	t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
	// Wait for the turn to finish (the CLI exited, the read loop is done).
	deadline := time.After(5 * time.Second)
	for done := false; !done; {
		select {
		case evt := <-s.Events():
			done = evt.Type == core.EventResult
		case <-deadline:
			t.Fatal("turn did not finish")
		}
	}
	if !alive(pid) {
		t.Fatal("precondition: descendant already gone")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	waitGone := time.Now().Add(2 * time.Second)
	for alive(pid) && time.Now().Before(waitGone) {
		time.Sleep(10 * time.Millisecond)
	}
	if alive(pid) {
		t.Fatal("descendant of an exited CLI survived Close")
	}
}

// Item 2: the CLI exits 0 and a descendant keeps stderr open silently:
// main's empty result, not an error.
func TestOpencodeCleanExitWithHeldStderrIsResult(t *testing.T) {
	script := `sleep 30 </dev/null >/dev/null &` + "\nexit 0"
	s, err := newOpencodeSession(context.Background(), writeFakeOpencode(t, script), nil, t.TempDir(), "", "default", "", "", nil)
	if err != nil {
		t.Fatalf("newOpencodeSession: %v", err)
	}
	defer func() { _ = s.Close() }()
	if err := s.Send("hi", "m1", nil, nil); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case evt := <-s.Events():
		if evt.Type != core.EventResult {
			t.Fatalf("event = %v (%v), want EventResult", evt.Type, evt.Error)
		}
	case <-time.After(pipeWaitDelay + 5*time.Second):
		t.Fatal("no event")
	}
}
