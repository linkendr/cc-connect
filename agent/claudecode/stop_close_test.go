package claudecode

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

// Fork regression tests for linkendr/cc-connect PR 7: /stop on a Claude Code
// session takes the engine's close path, as on the fork's `fallback` branch
// before the upstream sync. They run the real engine and the real claudecode
// adapter against this test binary as a fake `claude` CLI. The fake answers an
// interrupt the way Claude Code does (a terminating result, process kept
// alive), so a session that exposes CancelTurn to the engine keeps its process.

const (
	stopHelperEnv = "CC_CONNECT_CLAUDE_STOP_HELPER"
	// stopRoleEnv names the fake's prompt log ("primary" or "fallback").
	stopRoleEnv = "CC_CONNECT_CLAUDE_STOP_ROLE"
	// stopResultEnv selects the result that ends a stopped turn: "empty"
	// (the interrupt result) or "ratelimit" (a retriable backend error).
	stopResultEnv = "CC_CONNECT_CLAUDE_STOP_RESULT"
	// stopLateEnv makes the fake emit the stopped turn's terminal result
	// again after the next prompt reaches the same process.
	stopLateEnv = "CC_CONNECT_CLAUDE_STOP_LATE"

	stopSessionID = "stop-test-session"
)

func TestClaudeStopHelperProcess(t *testing.T) {
	if os.Getenv(stopHelperEnv) != "1" {
		return
	}
	appendLine := func(name, line string) {
		f, err := os.OpenFile(name, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			os.Exit(2)
		}
		fmt.Fprintln(f, line)
		f.Close()
	}
	out := json.NewEncoder(os.Stdout)
	emit := func(v map[string]any) {
		if err := out.Encode(v); err != nil {
			os.Exit(2)
		}
	}
	result := func(subtype string, isErr bool, text string, extra map[string]any) {
		ev := map[string]any{"type": "result", "subtype": subtype, "is_error": isErr, "session_id": stopSessionID}
		if text != "" {
			ev["result"] = text
		}
		for k, v := range extra {
			ev[k] = v
		}
		emit(ev)
	}
	stoppedResult := func() {
		if os.Getenv(stopResultEnv) == "ratelimit" {
			result("success", true, "API Error: 429 rate_limit_error", map[string]any{"api_error_status": 429})
			return
		}
		result("error_during_execution", true, "", nil)
	}

	role := os.Getenv(stopRoleEnv)
	appendLine("spawns.log", "start")
	active, interrupted := false, false
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)
	for in.Scan() {
		var msg struct {
			Type      string `json:"type"`
			RequestID string `json:"request_id"`
			Request   struct {
				Subtype string `json:"subtype"`
			} `json:"request"`
			Message struct {
				Content any `json:"content"`
			} `json:"message"`
		}
		if err := json.Unmarshal(in.Bytes(), &msg); err != nil {
			os.Exit(2)
		}
		switch msg.Type {
		case "user":
			prompt, _ := msg.Message.Content.(string)
			appendLine("prompts.log", role+" "+prompt)
			emit(map[string]any{"type": "system", "subtype": "init", "session_id": stopSessionID})
			if strings.HasPrefix(prompt, "hold") {
				active = true
				continue
			}
			if interrupted && os.Getenv(stopLateEnv) == "1" {
				result("success", false, "late-A", nil)
			}
			answer := "answer-" + prompt
			emit(map[string]any{"type": "assistant", "message": map[string]any{
				"content": []any{map[string]any{"type": "text", "text": answer}},
			}})
			result("success", false, answer, nil)
		case "control_request":
			if msg.Request.Subtype != "interrupt" {
				continue
			}
			emit(map[string]any{"type": "control_response", "response": map[string]any{
				"subtype": "success", "request_id": msg.RequestID,
			}})
			if active {
				active, interrupted = false, true
				stoppedResult()
			}
		}
	}
	// stdin EOF is how Close asks Claude Code to exit.
	if active {
		stoppedResult()
	}
	appendLine("spawns.log", "exit")
	os.Exit(0)
}

func stopHelperOpts(t *testing.T, workDir, role string, env map[string]string) map[string]any {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	merged := map[string]string{stopHelperEnv: "1", stopRoleEnv: role}
	for k, v := range env {
		merged[k] = v
	}
	return map[string]any{
		"work_dir":    workDir,
		"cc_data_dir": t.TempDir(),
		"cmd":         []string{exe, "-test.run=^TestClaudeStopHelperProcess$", "--"},
		"env":         merged,
	}
}

type stopTestPlatform struct {
	mu       sync.Mutex
	messages []string
}

func (p *stopTestPlatform) Name() string                    { return "stop-test" }
func (p *stopTestPlatform) Start(core.MessageHandler) error { return nil }
func (p *stopTestPlatform) Stop() error                     { return nil }
func (p *stopTestPlatform) Reply(_ context.Context, _ any, content string) error {
	p.record(content)
	return nil
}
func (p *stopTestPlatform) Send(_ context.Context, _ any, content string) error {
	p.record(content)
	return nil
}
func (p *stopTestPlatform) record(content string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.messages = append(p.messages, content)
}
func (p *stopTestPlatform) seen() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.messages...)
}

type stopHarness struct {
	t       *testing.T
	p       *stopTestPlatform
	engine  *core.Engine
	workDir string
	// fallbackDir is the fallback agent's work dir ("" without fallback).
	fallbackDir string
}

func newStopHarness(t *testing.T, env map[string]string, fallback bool) *stopHarness {
	t.Helper()
	h := &stopHarness{t: t, p: &stopTestPlatform{}, workDir: t.TempDir()}
	agent, err := New(stopHelperOpts(t, h.workDir, "primary", env))
	if err != nil {
		t.Fatal(err)
	}
	h.engine = core.NewEngine("stop-test", agent, []core.Platform{h.p}, "", core.LangEnglish)
	h.engine.SetStreamPreviewCfg(core.StreamPreviewCfg{Enabled: false})
	h.engine.SetShowContextIndicator(false)
	h.engine.SetShowWorkdirIndicator(false)
	h.engine.SetReplyFooterEnabled(false)
	if fallback {
		h.fallbackDir = t.TempDir()
		h.engine.SetAgentFallbacks([]core.AgentFallbackSpec{{
			Type:    "claudecode",
			Options: stopHelperOpts(t, h.fallbackDir, "fallback", env),
			Label:   "fallback",
		}})
	}
	t.Cleanup(func() {
		if err := h.engine.Stop(); err != nil {
			t.Error(err)
		}
	})
	return h
}

func (h *stopHarness) send(content string) {
	h.engine.ReceiveMessage(h.p, &core.Message{
		SessionKey: "stop-test:user", Platform: h.p.Name(), UserID: "user",
		MessageID: content, Content: content, ReplyCtx: content,
	})
}

// readLog returns a fake CLI log; a log not written yet reads as empty.
func readLog(t *testing.T, dir, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(b)
}

func (h *stopHarness) file(name string) string { return readLog(h.t, h.workDir, name) }

// waitFor polls cond until it holds; the fake CLI runs in another process, so
// its logs are the only signal of what it has received.
func (h *stopHarness) waitFor(what string, cond func() bool) {
	h.t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			h.t.Fatalf("timed out waiting for %s; spawns.log=%q prompts.log=%q platform=%q",
				what, h.file("spawns.log"), h.file("prompts.log"), h.p.seen())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (h *stopHarness) sawMessage(sub string) bool {
	for _, m := range h.p.seen() {
		if strings.Contains(m, sub) {
			return true
		}
	}
	return false
}

func (h *stopHarness) startHeldTurnAndStop() {
	h.t.Helper()
	h.send("hold-A")
	h.waitFor("turn A to reach the CLI", func() bool {
		return strings.Contains(h.file("prompts.log"), "primary hold-A")
	})
	h.send("/stop")
}

// On the PR base, /stop marked and removed the Claude state and closed the
// process; upstream's graceful CancelTurn kept the process. The fork keeps
// the base path.
func TestClaudeStop_ClosesProcessAndNextTurnStartsNewOne(t *testing.T) {
	if _, ok := any(&claudeSession{}).(core.AgentSessionCanceller); ok {
		t.Fatal("claudeSession implements core.AgentSessionCanceller; /stop would keep the process alive")
	}
	h := newStopHarness(t, nil, false)
	h.startHeldTurnAndStop()
	h.waitFor("the Claude process to exit after /stop", func() bool {
		return strings.Contains(h.file("spawns.log"), "exit")
	})

	h.send("B")
	h.waitFor("answer-B", func() bool { return h.sawMessage("answer-B") })
	if got := strings.Count(h.file("spawns.log"), "start"); got != 2 {
		t.Fatalf("Claude processes started = %d, want 2 (turn B needs a new process); spawns.log=%q", got, h.file("spawns.log"))
	}
}

// The PR 7 review interleaving: the stopped turn's terminal result reaches the
// event stream after the next turn's prompt. With the process kept alive that
// result finished turn B. On the close path turn B has its own process, so no
// event of turn A can reach it.
func TestClaudeStop_LateTerminalEventCannotReachNextTurn(t *testing.T) {
	h := newStopHarness(t, map[string]string{stopLateEnv: "1"}, false)
	h.startHeldTurnAndStop()
	h.send("B") // at once, without waiting for the old process to exit

	h.waitFor("answer-B", func() bool { return h.sawMessage("answer-B") })
	if h.sawMessage("late-A") {
		t.Fatalf("turn A's late result reached the user: %q", h.p.seen())
	}
}

// A retriable error that ends the stopped turn must not start the fallback
// chain: the user stopped that request.
func TestClaudeStop_StoppedTurnDoesNotStartFallback(t *testing.T) {
	h := newStopHarness(t, map[string]string{stopResultEnv: "ratelimit"}, true)
	h.startHeldTurnAndStop()
	h.waitFor("the Claude process to exit after /stop", func() bool {
		return strings.Contains(h.file("spawns.log"), "exit")
	})

	// The fake emitted turn A's error before it exited. Turn B, answered by
	// whichever agent, gives the engine time to act on that error.
	h.send("B")
	h.waitFor("answer-B", func() bool { return h.sawMessage("answer-B") })
	if fb := readLog(t, h.fallbackDir, "prompts.log"); strings.Contains(fb, "hold-A") {
		t.Fatalf("the stopped turn was sent to the fallback agent; fallback prompts.log=%q", fb)
	}
}
