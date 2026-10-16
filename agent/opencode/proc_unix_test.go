//go:build unix

package opencode

import (
	"context"
	"errors"
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

// A descendant that left the process group keeps stdout open: Close reports
// the timeout (so the engine never resumes into it), closes the pipes, and
// the read loop still exits.
func TestOpencodeCloseTimeoutClosesPipes(t *testing.T) {
	if _, err := exec.LookPath("perl"); err != nil {
		t.Skip("perl not available")
	}
	old := closeWaitTimeout
	closeWaitTimeout = 200 * time.Millisecond
	t.Cleanup(func() { closeWaitTimeout = old })

	// perl leaves the group first, then signals readiness with its PID.
	s, pid := startFakeTurn(t, `perl -e '$|=1; setpgrp(0,0); print "{\"type\":\"text\",\"part\":{\"text\":\"$$\"}}\n"; sleep 30' &`+"\nsleep 30")
	t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
	if err := s.Close(); err == nil {
		t.Fatal("Close returned nil although the read loop was still running")
	}
	exited := make(chan struct{})
	go func() { s.wg.Wait(); close(exited) }()
	select {
	case <-exited:
	case <-time.After(pipeWaitDelay + 2*time.Second):
		// exec closes the pipes cmd.WaitDelay after the cancel, so the read
		// loop exits although the stray descendant keeps them open.
		t.Fatal("read loop goroutine did not exit after the pipes were closed")
	}
	if errors.Is(syscall.Kill(pid, 0), syscall.ESRCH) {
		t.Log("escaped descendant already gone")
	}
}
