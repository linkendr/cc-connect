//go:build unix

package opencode

import (
	"context"
	"os/exec"
	"runtime"
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

// (b) The CLI exits 0 leaving a same-group descendant that holds no pipe:
// the turn kills the group before reaping the CLI, so the descendant is
// dead when the result is emitted; a process group created after the turn
// is never signalled by Close.
func TestOpencodeTurnEndKillsGroupBeforeReap(t *testing.T) {
	script := `sleep 30 </dev/null >/dev/null 2>&1 &` + "\n" +
		`echo "{\"type\":\"text\",\"part\":{\"text\":\"$!\"}}"` + "\nexit 0"
	s, pid := startFakeTurn(t, script)
	t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
	deadline := time.After(5 * time.Second)
	for done := false; !done; {
		select {
		case evt := <-s.Events():
			done = evt.Type == core.EventResult
		case <-deadline:
			t.Fatal("turn did not finish")
		}
	}
	// SIGKILL was sent before the result; allow the kernel a moment to
	// finish the kill and the zombie to be reaped by init.
	gone := time.Now().Add(time.Second)
	for alive(pid) && time.Now().Before(gone) {
		time.Sleep(5 * time.Millisecond)
	}
	if alive(pid) {
		t.Fatal("descendant still alive after the turn ended")
	}

	sentinel := exec.Command("sleep", "30")
	sentinel.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := sentinel.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sentinel.Process.Kill(); _ = sentinel.Wait() })
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !alive(sentinel.Process.Pid) {
		t.Fatal("Close signalled a process group created after the turn")
	}
}

// Item 2 (round 10): a descendant leaves the group, redirects stdout and
// holds stderr; the CLI exits 0. At the stderr deadline the turn closes
// its stderr read end and joins the reader, so the result arrives, all
// reader goroutines exit, and Close returns at once.
func TestOpencodeHeldStderrReaderJoinedAtDeadline(t *testing.T) {
	if _, err := exec.LookPath("perl"); err != nil {
		t.Skip("perl not available")
	}
	oldDelay, oldClose := pipeWaitDelay, closeWaitTimeout
	pipeWaitDelay, closeWaitTimeout = 200*time.Millisecond, 500*time.Millisecond
	t.Cleanup(func() { pipeWaitDelay, closeWaitTimeout = oldDelay, oldClose })

	s, pid := startFakeTurn(t, `perl -e '$|=1; setpgrp(0,0); print "{\"type\":\"text\",\"part\":{\"text\":\"$$\"}}\n"; open(STDOUT, ">/dev/null"); sleep 30' &`+"\nexit 0")
	t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
	select {
	case evt := <-s.Events():
		if evt.Type != core.EventResult {
			t.Fatalf("event = %v, want EventResult", evt.Type)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no result: the turn waited on the held stderr")
	}
	exited := make(chan struct{})
	go func() { s.wg.Wait(); close(exited) }()
	select {
	case <-exited:
	case <-time.After(time.Second):
		t.Fatal("stderr reader still running after the turn ended")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
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

// cliPID returns the PID of the session's running turn.
func cliPID(t *testing.T, s *opencodeSession) int {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		s.procMu.Lock()
		for p := range s.procs {
			if p.cmd.Process != nil {
				pid := p.cmd.Process.Pid
				s.procMu.Unlock()
				return pid
			}
		}
		s.procMu.Unlock()
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("no running turn")
	return 0
}

func firstEvent(t *testing.T, s *opencodeSession) core.Event {
	t.Helper()
	select {
	case evt := <-s.Events():
		return evt
	case <-time.After(pipeWaitDelay + 5*time.Second):
		t.Fatal("no event")
		return core.Event{}
	}
}

func needsLeaderState(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("leader exit state not readable on " + runtime.GOOS)
	}
}

// Round 11 (1): the CLI is SIGKILLed by someone else before any output:
// a provider-side error, so the fallback chain advances.
func TestOpencodeExternalSIGKILLIsError(t *testing.T) {
	needsLeaderState(t)
	s, err := newOpencodeSession(context.Background(), writeFakeOpencode(t, "exec sleep 30"), nil, t.TempDir(), "", "default", "", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.Send("hi", "m1", nil, nil); err != nil {
		t.Fatal(err)
	}
	pid := cliPID(t, s)
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	evt := firstEvent(t, s)
	if evt.Type != core.EventError || evt.ErrorKind != core.ErrorKindUnknown {
		t.Fatalf("event = %v kind=%q, want provider-side EventError", evt.Type, evt.ErrorKind)
	}
}

// Round 11 (2): our own turn-end kill of a CLI that closed its output but
// kept running: unchanged, an empty result (no silent-crash error).
func TestOpencodeOwnTurnEndKillIsResult(t *testing.T) {
	needsLeaderState(t)
	s, err := newOpencodeSession(context.Background(), writeFakeOpencode(t, "exec sleep 30 >&- 2>&-"), nil, t.TempDir(), "", "default", "", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.Send("hi", "m1", nil, nil); err != nil {
		t.Fatal(err)
	}
	if evt := firstEvent(t, s); evt.Type != core.EventResult {
		t.Fatalf("event = %v (%v), want EventResult", evt.Type, evt.Error)
	}
}
