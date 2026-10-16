package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ---------- scripted agents (pattern from upstream PR #557) ----------

// scriptedTurn is what a scripted session emits for one Send.
type scriptedTurn []Event

func okTurn(text string) scriptedTurn {
	return scriptedTurn{{Type: EventResult, Content: text, Done: true}}
}

// usageLimitTurn mimics Claude Code's subscription cap: an EventResult whose
// content is the limit text, classified by the adapter as usage_limit.
func usageLimitTurn(resetAt time.Time) scriptedTurn {
	return scriptedTurn{{
		Type:         EventResult,
		Content:      "You've hit your session limit · resets 3pm (America/Los_Angeles)",
		Done:         true,
		ErrorKind:    ErrorKindUsageLimit,
		ErrorResetAt: resetAt,
	}}
}

func errorTurn(msg string, kind ErrorKind) scriptedTurn {
	return scriptedTurn{{Type: EventError, Error: errors.New(msg), ErrorKind: kind}}
}

// queuingAgentSession (engine_test.go) records prompts; scriptedSession adds
// a reply script: the i-th Send emits turns[i].
type scriptedSession struct {
	*queuingAgentSession
	mu     sync.Mutex
	turns  []scriptedTurn
	sends  int
	closed bool
	dead   bool // Alive()=false without closing the channel (process exited)

	// Optional test hooks.
	beforeTurn func(n int)        // runs before turn n is emitted (inside Send, unlocked)
	afterTurn  func(n int)        // runs after turn n was emitted (inside Send, unlocked)
	dieAfter   map[int]bool       // mark dead after emitting turn n
	closeAfter map[int]bool       // close the event channel after emitting turn n
	onAllow    scriptedTurn       // emitted when a permission is allowed
	perms      []PermissionResult // RespondPermission calls
	aliveCalls atomic.Int32       // Alive() call count
}

func newScriptedSession(id string, turns ...scriptedTurn) *scriptedSession {
	return &scriptedSession{queuingAgentSession: newQueuingSession(id), turns: turns}
}

func (s *scriptedSession) Send(prompt string, messageID string, images []ImageAttachment, files []FileAttachment) error {
	_ = s.queuingAgentSession.Send(prompt, messageID, images, files)
	s.mu.Lock()
	n := s.sends
	s.sends++
	hook := s.beforeTurn
	s.mu.Unlock()
	if hook != nil {
		hook(n)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || n >= len(s.turns) {
		return nil
	}
	for _, ev := range s.turns[n] {
		s.events <- ev // buffered (16); scripts are short
	}
	if s.dieAfter[n] {
		s.dead = true
	}
	if s.closeAfter[n] {
		s.closed = true
		_ = s.queuingAgentSession.Close()
	}
	after := s.afterTurn
	s.mu.Unlock()
	if after != nil {
		after(n)
	}
	s.mu.Lock() // re-taken for the deferred Unlock
	return nil
}

func (s *scriptedSession) RespondPermission(_ string, res PermissionResult) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.perms = append(s.perms, res)
	if res.Behavior == "allow" && !s.closed {
		for _, ev := range s.onAllow {
			s.events <- ev
		}
	}
	return nil
}

func (s *scriptedSession) permissions() []PermissionResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]PermissionResult(nil), s.perms...)
}

func (s *scriptedSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.queuingAgentSession.Close()
}

// Alive reads the mutex-guarded flag; the embedded stub's plain bool would
// race with a Close from the engine's async teardown.
func (s *scriptedSession) Alive() bool {
	s.aliveCalls.Add(1)
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.closed && !s.dead
}

func (s *scriptedSession) prompts() []string {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	return append([]string(nil), s.sendCalls...)
}

// scriptedAgent hands out its scripted session; after a restart it hands
// out the next one from more (the last one is reused).
type scriptedAgent struct {
	name    string
	display string
	session *scriptedSession
	more    []*scriptedSession
	onStart func() // runs inside StartSession, unlocked
	mu      sync.Mutex
	starts  int
}

func (a *scriptedAgent) Name() string        { return a.name }
func (a *scriptedAgent) DisplayName() string { return a.display }
func (a *scriptedAgent) StartSession(_ context.Context, _ string) (AgentSession, error) {
	a.mu.Lock()
	hook := a.onStart
	a.mu.Unlock()
	if hook != nil {
		hook()
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.starts++
	if a.starts > 1 && len(a.more) > 0 {
		a.session = a.more[0]
		a.more = a.more[1:]
	}
	return a.session, nil
}
func (a *scriptedAgent) ListSessions(_ context.Context) ([]AgentSessionInfo, error) { return nil, nil }
func (a *scriptedAgent) Stop() error                                                { return nil }
func (a *scriptedAgent) StartCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.starts
}

// ---------- harness ----------

// hookPlatform is a stub platform that can react synchronously to what the
// engine sends (e.g. issue /stop right after the fallback notice).
type hookPlatform struct {
	stubPlatformEngine
	hookMu       sync.Mutex
	onSend       func(content string)
	onTyping     func()
	onTypingStop func()
}

func (h *hookPlatform) Send(ctx context.Context, rc any, content string) error {
	_ = h.stubPlatformEngine.Send(ctx, rc, content)
	h.hookMu.Lock()
	fn := h.onSend
	h.hookMu.Unlock()
	if fn != nil {
		fn(content)
	}
	return nil
}

func (h *hookPlatform) Reply(ctx context.Context, rc any, content string) error {
	return h.Send(ctx, rc, content)
}

// StartTyping makes hookPlatform a TypingIndicator; onTyping (optional) runs
// inside StartTyping, and the returned stop func runs onTypingStop.
func (h *hookPlatform) StartTyping(_ context.Context, _ any) func() {
	h.hookMu.Lock()
	start, stop := h.onTyping, h.onTypingStop
	h.hookMu.Unlock()
	if start != nil {
		start()
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			if stop != nil {
				stop()
			}
		})
	}
}

func (h *hookPlatform) setTypingHooks(start, stop func()) {
	h.hookMu.Lock()
	h.onTyping, h.onTypingStop = start, stop
	h.hookMu.Unlock()
}

type fallbackEnv struct {
	t        *testing.T
	e        *Engine
	p        *hookPlatform
	primary  *scriptedAgent
	fbs      []*scriptedAgent
	built    []int // factory calls per fallback index
	mu       sync.Mutex
	now      time.Time
	msgCount int
}

func newFallbackEnv(t *testing.T, primaryTurns []scriptedTurn, fallbackTurns ...[]scriptedTurn) *fallbackEnv {
	t.Helper()
	env := &fallbackEnv{
		t:       t,
		p:       &hookPlatform{stubPlatformEngine: stubPlatformEngine{n: "test"}},
		primary: &scriptedAgent{name: "primary", display: "Claude Code", session: newScriptedSession("primary-sid", primaryTurns...)},
		now:     time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC),
	}
	env.e = NewEngine("test", env.primary, []Platform{env.p}, "", LangEnglish)
	t.Cleanup(func() { _ = env.e.Stop() })
	if len(fallbackTurns) == 0 {
		return env
	}
	specs := make([]AgentFallbackSpec, len(fallbackTurns))
	env.built = make([]int, len(fallbackTurns))
	for i, turns := range fallbackTurns {
		env.fbs = append(env.fbs, &scriptedAgent{
			name:    fmt.Sprintf("fb%d", i),
			display: "OpenCode",
			session: newScriptedSession(fmt.Sprintf("fb%d-sid", i), turns...),
		})
		specs[i] = AgentFallbackSpec{Type: fmt.Sprintf("fb%d", i), Options: map[string]any{"model": fmt.Sprintf("model-%d", i)}}
	}
	env.e.SetAgentFallbacks(specs)
	env.e.agentFallback.factory = func(name string, opts map[string]any) (Agent, error) {
		env.mu.Lock()
		defer env.mu.Unlock()
		for i, a := range env.fbs {
			if a.name == name {
				env.built[i]++
				return a, nil
			}
		}
		return nil, fmt.Errorf("unknown agent %q", name)
	}
	env.e.agentFallback.now = func() time.Time {
		env.mu.Lock()
		defer env.mu.Unlock()
		return env.now
	}
	return env
}

func (env *fallbackEnv) advance(d time.Duration) {
	env.mu.Lock()
	env.now = env.now.Add(d)
	env.mu.Unlock()
}

func (env *fallbackEnv) builtCount(i int) int {
	env.mu.Lock()
	defer env.mu.Unlock()
	return env.built[i]
}

const fallbackTestKey = "test:user1"

// send delivers one user message and waits until the turn has finished and
// released the session lock.
func (env *fallbackEnv) send(content string) {
	env.t.Helper()
	env.msgCount++
	env.e.ReceiveMessage(env.p, &Message{
		SessionKey: fallbackTestKey,
		Platform:   "test",
		MessageID:  fmt.Sprintf("m%d", env.msgCount),
		UserID:     "u1",
		UserName:   "user",
		Content:    content,
		ReplyCtx:   "ctx",
	})
	env.waitIdle()
}

func (env *fallbackEnv) waitIdle() {
	env.t.Helper()
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		// handleMessage takes the session lock before it starts the turn
		// goroutine, so a free lock means the turn has finished.
		if gen, ok := session.TryLock(); ok {
			session.Unlock(gen)
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	env.t.Fatalf("turn did not finish; sent = %#v", env.p.getSent())
}

// waitSent polls until some sent message contains want.
func (env *fallbackEnv) waitSent(want string) {
	env.t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, s := range env.p.getSent() {
			if strings.Contains(s, want) {
				return
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	env.t.Fatalf("no sent message contains %q; sent = %#v", want, env.p.getSent())
}

func (env *fallbackEnv) sentContaining(sub string) int {
	n := 0
	for _, s := range env.p.getSent() {
		if strings.Contains(s, sub) {
			n++
		}
	}
	return n
}

// ---------- tests ----------

// (a) Primary success: no fallback agent is built or used.
func TestAgentFallback_PrimarySuccessNoFallback(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{okTurn("primary answer")},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	env.send("hello")
	env.waitSent("primary answer")

	if n := env.builtCount(0); n != 0 {
		t.Errorf("fallback agent built %d times, want 0", n)
	}
	if got := env.sentContaining("answering with"); got != 0 {
		t.Errorf("fallback notice sent %d times, want 0; sent = %#v", got, env.p.getSent())
	}
}

// (b) Retriable primary failure: fallback[0] answers the same message after a
// one-line notice, gets the earlier conversation as context, and the next
// message goes to the primary again.
func TestAgentFallback_RetriableErrorUsesFallbackThenPrimaryAgain(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{
			okTurn("first primary answer"),
			usageLimitTurn(time.Time{}), // no reset time: next message tries the primary
			okTurn("third primary answer"),
		},
		[]scriptedTurn{okTurn("fallback answer")},
	)

	env.send("first question")
	env.waitSent("first primary answer")

	env.send("second question")
	env.waitSent("fallback answer")

	const notice = "Claude Code hit its usage limit; answering with OpenCode (model-0)"
	if got := env.sentContaining(notice); got != 1 {
		t.Errorf("notice %q sent %d times, want 1; sent = %#v", notice, got, env.p.getSent())
	}
	if got := env.sentContaining("You've hit your session limit"); got != 0 {
		t.Errorf("primary limit text leaked to the user: %#v", env.p.getSent())
	}

	fbPrompts := env.fbs[0].session.prompts()
	if len(fbPrompts) != 1 {
		t.Fatalf("fallback got %d prompts, want 1", len(fbPrompts))
	}
	p := fbPrompts[0]
	for _, want := range []string{"[Context:", "User: first question", "Assistant: first primary answer", "second question"} {
		if !strings.Contains(p, want) {
			t.Errorf("fallback prompt missing %q:\n%s", want, p)
		}
	}
	if strings.Contains(p, "User: second question") {
		t.Errorf("current message duplicated into the context block:\n%s", p)
	}

	// The primary's stored agent session ID (its --resume target) is untouched.
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	if got := session.GetAgentSessionID(); got != "primary-sid" {
		t.Errorf("primary agent session ID = %q, want primary-sid", got)
	}
	// The fallback answer is part of cc-connect's history.
	hist := session.GetHistory(0)
	if last := hist[len(hist)-1]; last.Role != "assistant" || last.Content != "fallback answer" {
		t.Errorf("last history entry = %+v, want fallback answer", last)
	}

	// Turn-scoped: the next message starts on the primary again.
	env.send("third question")
	env.waitSent("third primary answer")
	if got := len(env.primary.session.prompts()); got != 3 {
		t.Errorf("primary got %d prompts, want 3", got)
	}
	if got := len(env.fbs[0].session.prompts()); got != 1 {
		t.Errorf("fallback got %d prompts, want 1", got)
	}
}

// (b2) A classified stderr-style EventError (claude exited with
// rate_limit_error) also triggers the fallback.
func TestAgentFallback_RetriableEventErrorUsesFallback(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{errorTurn(`API Error: {"error":{"type":"rate_limit_error"}}`, ErrorKindRateLimit)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	env.send("question")
	env.waitSent("fallback answer")
	if got := env.sentContaining("Claude Code is rate limited; answering with OpenCode (model-0)"); got != 1 {
		t.Errorf("rate-limit notice count = %d; sent = %#v", got, env.p.getSent())
	}
	if got := env.sentContaining("rate_limit_error"); got != 0 {
		t.Errorf("primary error leaked to the user: %#v", env.p.getSent())
	}
}

// (c) Primary and every fallback fail: the last error is shown, after one
// notice per hop.
func TestAgentFallback_AllFailSurfacesError(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{errorTurn("openai quota exceeded", ErrorKindUnknown)},
		[]scriptedTurn{errorTurn("deepseek is down", ErrorKindUnknown)},
	)
	env.send("question")
	env.waitSent("deepseek is down")

	sent := env.p.getSent()
	if got := env.sentContaining("Claude Code hit its usage limit; answering with OpenCode (model-0)"); got != 1 {
		t.Errorf("first notice count = %d; sent = %#v", got, sent)
	}
	if got := env.sentContaining("OpenCode (model-0) failed; answering with OpenCode (model-1)"); got != 1 {
		t.Errorf("second notice count = %d; sent = %#v", got, sent)
	}
	// The intermediate fallback error is not shown; only the final one is.
	if got := env.sentContaining("openai quota exceeded"); got != 0 {
		t.Errorf("intermediate fallback error shown: %#v", sent)
	}
}

// (d) A non-retriable primary error is shown as before; no fallback.
func TestAgentFallback_NonRetriableErrorNoFallback(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{errorTurn("compilation failed", ErrorKindUnknown)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	env.send("question")
	env.waitSent("compilation failed")
	if n := env.builtCount(0); n != 0 {
		t.Errorf("fallback agent built %d times, want 0", n)
	}
	if got := env.sentContaining("answering with"); got != 0 {
		t.Errorf("unexpected fallback notice; sent = %#v", env.p.getSent())
	}
}

// (e) A reset time from the primary blocks it: later messages go straight to
// fallback[0] until the reset passes, then the primary answers again.
func TestAgentFallback_BlockedUntilSkipsPrimary(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{
			usageLimitTurn(time.Date(2026, 10, 16, 13, 0, 0, 0, time.UTC)), // env.now + 1h
			okTurn("primary is back"),
		},
		[]scriptedTurn{okTurn("fallback one"), okTurn("fallback two")},
	)

	env.send("first")
	env.waitSent("fallback one")

	env.advance(30 * time.Minute)
	env.send("second")
	env.waitSent("fallback two")
	if got := len(env.primary.session.prompts()); got != 1 {
		t.Fatalf("primary got %d prompts while blocked, want 1", got)
	}
	if got := env.sentContaining("Claude Code is at its usage limit until"); got != 1 {
		t.Errorf("blocked notice count = %d; sent = %#v", got, env.p.getSent())
	}
	// The second fallback turn only gets context it has not seen yet.
	fbPrompts := env.fbs[0].session.prompts()
	if len(fbPrompts) != 2 {
		t.Fatalf("fallback got %d prompts, want 2", len(fbPrompts))
	}
	if strings.Contains(fbPrompts[1], "User: first") {
		t.Errorf("second hand-off repeats context the fallback already has:\n%s", fbPrompts[1])
	}

	env.advance(31 * time.Minute) // past the reset
	env.send("third")
	env.waitSent("primary is back")
	if got := len(env.primary.session.prompts()); got != 2 {
		t.Errorf("primary got %d prompts after reset, want 2", got)
	}
}

// No fallback configured: a usage-limit result is shown exactly as before.
func TestAgentFallback_EmptyChainKeepsExistingBehavior(t *testing.T) {
	env := newFallbackEnv(t, []scriptedTurn{usageLimitTurn(time.Now().Add(time.Hour))})
	env.send("question")
	env.waitSent("You've hit your session limit")
	if env.e.agentFallback != nil {
		t.Fatal("agentFallback set without SetAgentFallbacks")
	}
	if got := env.sentContaining("answering with"); got != 0 {
		t.Errorf("unexpected fallback notice; sent = %#v", env.p.getSent())
	}
}

// A fallback agent that cannot be built is skipped; the next one answers.
func TestAgentFallback_BuildErrorSkipsToNext(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("never")},
		[]scriptedTurn{okTurn("second fallback answer")},
	)
	env.e.agentFallback.specs[0].Type = "missing-cli"
	env.send("question")
	env.waitSent("second fallback answer")
	if got := env.sentContaining("answering with OpenCode (model-1)"); got != 1 {
		t.Errorf("notice for fallback[1] count = %d; sent = %#v", got, env.p.getSent())
	}
}

func TestAgentFallbackChain_RecordPrimaryFailure(t *testing.T) {
	now := time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name        string
		kind        ErrorKind
		resetAt     time.Time
		wantBlocked bool
		wantUntil   time.Time
	}{
		{"no reset time", ErrorKindUsageLimit, time.Time{}, false, time.Time{}},
		{"reset in the past", ErrorKindUsageLimit, now.Add(-time.Minute), false, time.Time{}},
		{"usage limit with reset", ErrorKindUsageLimit, now.Add(2 * time.Hour), true, now.Add(2 * time.Hour)},
		{"rate limit retry-after", ErrorKindRateLimit, now.Add(30 * time.Second), true, now.Add(30 * time.Second)},
		{"overloaded never blocks", ErrorKindOverloaded, now.Add(time.Hour), false, time.Time{}},
		{"far future is capped", ErrorKindUsageLimit, now.Add(400 * 24 * time.Hour), true, now.Add(maxPrimaryBlock)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newAgentFallbackChain([]AgentFallbackSpec{{Type: "x"}})
			c.now = func() time.Time { return now }
			c.recordPrimaryFailure(tc.kind, tc.resetAt)
			until, kind, blocked := c.primaryBlocked()
			if blocked != tc.wantBlocked || !until.Equal(tc.wantUntil) {
				t.Fatalf("primaryBlocked = (%v, %q, %v), want (%v, _, %v)", until, kind, blocked, tc.wantUntil, tc.wantBlocked)
			}
			if blocked && kind != tc.kind {
				t.Errorf("kind = %q, want %q", kind, tc.kind)
			}
		})
	}
}

func TestFormatResetTime(t *testing.T) {
	now := time.Date(2026, 10, 16, 9, 0, 0, 0, time.Local)
	if got := formatResetTime(now.Add(6*time.Hour), now); got != now.Add(6*time.Hour).Format("15:04 MST") {
		t.Errorf("same day = %q", got)
	}
	later := time.Date(2026, 10, 20, 9, 30, 0, 0, time.Local)
	if got, want := formatResetTime(later, now), later.Format("Jan 2 15:04 MST"); got != want {
		t.Errorf("other day = %q, want %q", got, want)
	}
}

// /stop on the user's session also stops a fallback agent mid-answer.
func TestAgentFallback_StopAlsoStopsFallbackTurn(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{{}}, // fallback never answers
	)
	env.e.ReceiveMessage(env.p, &Message{SessionKey: fallbackTestKey, Platform: "test", MessageID: "m1", UserID: "u1", Content: "question", ReplyCtx: "ctx"})
	env.waitSent("answering with")

	fbKey := fallbackStateKey(fallbackTestKey, 0)
	deadline := time.Now().Add(3 * time.Second)
	for {
		env.e.interactiveMu.Lock()
		_, running := env.e.interactiveStates[fbKey]
		env.e.interactiveMu.Unlock()
		if running {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("fallback state never started")
		}
		time.Sleep(5 * time.Millisecond)
	}

	if !env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") {
		t.Fatal("stopInteractiveSession returned false")
	}
	env.e.interactiveMu.Lock()
	_, still := env.e.interactiveStates[fbKey]
	env.e.interactiveMu.Unlock()
	if still {
		t.Error("fallback state still registered after /stop")
	}
	select {
	case <-env.fbs[0].session.controllableAgentSession.closed:
	case <-time.After(3 * time.Second):
		t.Error("fallback agent session was not closed by /stop")
	}
}

// ---------- review fixes (GPT-6 Astra findings) ----------

// receive delivers a message without waiting for its turn.
func (env *fallbackEnv) receive(content string) {
	env.msgCount++
	env.e.ReceiveMessage(env.p, &Message{
		SessionKey: fallbackTestKey,
		Platform:   "test",
		MessageID:  fmt.Sprintf("m%d", env.msgCount),
		UserID:     "u1",
		UserName:   "user",
		Content:    content,
		ReplyCtx:   "ctx",
	})
}

// gate blocks a scripted session before turn n until release is closed;
// entered is closed when the turn reaches the gate.
func gate(s *scriptedSession, n int) (entered, release chan struct{}) {
	entered, release = make(chan struct{}), make(chan struct{})
	s.mu.Lock()
	s.beforeTurn = func(i int) {
		if i == n {
			close(entered)
			<-release
		}
	}
	s.mu.Unlock()
	return entered, release
}

func waitClosed(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

// waitQueued waits until the primary state holds n queued messages.
func (env *fallbackEnv) waitQueued(n int) {
	env.t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		env.e.interactiveMu.Lock()
		st := env.e.interactiveStates[fallbackTestKey]
		env.e.interactiveMu.Unlock()
		if st != nil {
			st.mu.Lock()
			q := len(st.pendingMessages)
			st.mu.Unlock()
			if q >= n {
				return
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	env.t.Fatalf("message was not queued; sent = %#v", env.p.getSent())
}

// P1-1: a message queued during a fallback turn is drained on the primary
// with exclusive ownership of its event channel: no unsolicited reader runs
// while it is sent, its permission request reaches the user (not
// auto-denied), and it gets its own result.
func TestAgentFallback_QueuedTurnAfterFallbackOwnsEventChannel(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{
			usageLimitTurn(time.Time{}),
			{{Type: EventPermissionRequest, RequestID: "r1", ToolName: "Bash", ToolInput: "ls", ToolInputRaw: map[string]any{"command": "ls"}}},
		},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	primary := env.primary.session
	primary.onAllow = okTurn("queued answer")
	readerRunningAtSend := make(chan bool, 1)
	primary.beforeTurn = func(n int) {
		if n != 1 {
			return
		}
		env.e.interactiveMu.Lock()
		st := env.e.interactiveStates[fallbackTestKey]
		env.e.interactiveMu.Unlock()
		st.mu.Lock()
		readerRunningAtSend <- st.unsolicitedCancel != nil
		st.mu.Unlock()
	}
	entered, release := gate(env.fbs[0].session, 0)

	env.receive("first")
	waitClosed(t, entered, "fallback turn")
	env.receive("second") // queued behind the fallback turn
	env.waitQueued(1)
	close(release)

	select {
	case running := <-readerRunningAtSend:
		if running {
			t.Fatal("unsolicited reader was running when the queued turn was sent")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("queued message never reached the primary")
	}
	// The permission request is the queued turn's own: the user answers it.
	deadline := time.Now().Add(3 * time.Second)
	for {
		if _, pending := env.e.lookupPending(fallbackTestKey); pending != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("permission request never reached the foreground turn; perms = %+v", primary.permissions())
		}
		time.Sleep(5 * time.Millisecond)
	}
	env.receive("allow")
	env.waitSent("queued answer")
	env.waitIdle()
	for _, p := range primary.permissions() {
		if p.Behavior != "allow" {
			t.Errorf("permission auto-answered %+v", p)
		}
	}
}

// P1-2: /stop between the fallback notice and the fallback send cancels the
// turn: the fallback agent is never sent the message.
func TestAgentFallback_StopAfterNoticeCancelsFallback(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("must not run")},
	)
	var once sync.Once
	stopped := make(chan struct{})
	env.p.hookMu.Lock()
	env.p.onSend = func(content string) {
		if strings.Contains(content, "answering with") {
			once.Do(func() {
				env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx")
				close(stopped)
			})
		}
	}
	env.p.hookMu.Unlock()
	sent := make(chan struct{})
	env.fbs[0].session.mu.Lock()
	env.fbs[0].session.beforeTurn = func(int) { close(sent) }
	env.fbs[0].session.mu.Unlock()

	cancelled := watchLog(t, "agent fallback: turn cancelled")
	env.receive("question")
	waitClosed(t, stopped, "/stop after notice")
	waitClosed(t, cancelled, "fallback cancellation")
	select {
	case <-sent:
		t.Fatal("fallback agent was sent the stopped message")
	default:
	}
	env.e.interactiveMu.Lock()
	_, registered := env.e.interactiveStates[fallbackStateKey(fallbackTestKey, 0)]
	env.e.interactiveMu.Unlock()
	if registered {
		t.Error("fallback state left registered after a cancelled turn")
	}
}

// P1-4: the last fallback exits without text or an error event: the user
// gets an error, not just the notice.
func TestAgentFallback_FinalFallbackSilentExitShowsError(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{{}},
	)
	env.fbs[0].session.closeAfter = map[int]bool{0: true}
	env.send("question")
	env.waitSent("fallback agent exited without an answer")
}

// P2-5: a fresh chat during a known block sends its first message straight
// to the fallback; a message queued behind it must also skip the blocked
// primary.
func TestAgentFallback_FreshChatDuringBlockQueuedMessageSkipsPrimary(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{okTurn("primary must not answer")},
		[]scriptedTurn{okTurn("fb one"), okTurn("fb two")},
	)
	env.e.agentFallback.recordPrimaryFailure(ErrorKindUsageLimit, env.now.Add(time.Hour))
	entered, release := gate(env.fbs[0].session, 0)

	env.receive("first")
	waitClosed(t, entered, "fallback turn")
	env.receive("second")
	env.waitQueued(1)
	close(release)
	env.waitSent("fb two")
	env.waitIdle()
	if got := len(env.primary.session.prompts()); got != 0 {
		t.Errorf("blocked primary got %d prompts, want 0", got)
	}
}

// P2-6: the primary dies on a retriable process-exit error with no reset
// time while a message is queued; the queued message restarts the primary
// instead of being dropped with "agent session ended".
func TestAgentFallback_DeadPrimaryQueuedMessageRestartsPrimary(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{errorTurn(`API Error: {"error":{"type":"rate_limit_error"}}`, ErrorKindRateLimit)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	first := env.primary.session
	first.dieAfter = map[int]bool{0: true}
	second := newScriptedSession("primary-sid-2", okTurn("queued primary answer"))
	env.primary.more = []*scriptedSession{second}
	entered, release := gate(first, 0)

	env.receive("first")
	waitClosed(t, entered, "primary turn")
	env.receive("second") // queued while the primary is still alive
	env.waitQueued(1)
	close(release)

	env.waitSent("fallback answer")
	env.waitSent("queued primary answer")
	env.waitIdle()
	if got := env.sentContaining("agent session ended"); got != 0 {
		t.Errorf("queued message dropped: %#v", env.p.getSent())
	}
	if got := env.primary.StartCount(); got != 2 {
		t.Errorf("primary started %d times, want 2 (restart for the queued message)", got)
	}
	if got := second.prompts(); len(got) != 1 || !strings.Contains(got[0], "second") {
		t.Errorf("restarted primary prompts = %q", got)
	}
}

// P2-7: hooks see the user's conversation key, never the internal
// "#fallback:N" state key, and no session start/end for fallback states.
func TestAgentFallback_HooksSeeConversationKey(t *testing.T) {
	var mu sync.Mutex
	var got []HookEvent
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var ev HookEvent
		_ = json.NewDecoder(r.Body).Decode(&ev)
		mu.Lock()
		got = append(got, ev)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{errorTurn("openai quota exceeded", ErrorKindUnknown)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	async := false
	env.e.SetHooks(NewHookManager("test", []HookConfig{{Event: "*", Type: "http", URL: srv.URL, Async: &async}}, "sh", "-c", ""))
	env.send("question")
	env.waitSent("fallback answer")
	env.waitIdle()

	mu.Lock()
	defer mu.Unlock()
	sawSent := false
	for _, ev := range got {
		if ev.SessionKey != fallbackTestKey {
			t.Errorf("hook %s has session key %q, want %q", ev.Event, ev.SessionKey, fallbackTestKey)
		}
		if ev.Event == HookEventMessageSent && strings.Contains(ev.Content, "fallback answer") {
			sawSent = true
		}
	}
	if !sawSent {
		t.Errorf("no message.sent hook for the fallback answer; got %+v", got)
	}
	starts := 0
	for _, ev := range got {
		if ev.Event == HookEventSessionStarted {
			starts++
		}
	}
	if starts != 1 {
		t.Errorf("session.started hooks = %d, want 1 (primary only)", starts)
	}
}

// ---------- review round 2 ----------

// logWatcher is a slog handler that closes ch on the first record whose
// message contains substr; other records go to the previous handler.
type logWatcher struct {
	next   slog.Handler
	substr string
	once   *sync.Once
	ch     chan struct{}
}

func (w logWatcher) Enabled(ctx context.Context, l slog.Level) bool { return true }
func (w logWatcher) Handle(ctx context.Context, r slog.Record) error {
	if strings.Contains(r.Message, w.substr) {
		w.once.Do(func() { close(w.ch) })
	}
	if w.next.Enabled(ctx, r.Level) {
		return w.next.Handle(ctx, r)
	}
	return nil
}
func (w logWatcher) WithAttrs(a []slog.Attr) slog.Handler {
	w.next = w.next.WithAttrs(a)
	return w
}
func (w logWatcher) WithGroup(n string) slog.Handler {
	w.next = w.next.WithGroup(n)
	return w
}

// watchLog returns a channel closed when the engine logs a message
// containing substr: a deterministic "this code path ran" signal.
func watchLog(t *testing.T, substr string) <-chan struct{} {
	t.Helper()
	prev := slog.Default()
	ch := make(chan struct{})
	// Not prev.Handler(): the default handler writes through package log,
	// which SetDefault redirects back into slog (recursion).
	next := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})
	slog.SetDefault(slog.New(logWatcher{next: next, substr: substr, once: &sync.Once{}, ch: ch}))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return ch
}

// drainFixture builds a primary interactive state with one queued message,
// as the engine has it when drainPendingMessages starts, and locks the
// session for the turn.
func (env *fallbackEnv) drainFixture(primary *scriptedSession, withFallbackPolicy bool) (*interactiveState, *Session, uint64) {
	env.t.Helper()
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	gen, ok := session.TryLock()
	if !ok {
		env.t.Fatal("session already locked")
	}
	st := &interactiveState{
		agentSession: primary,
		busySession:  session,
		platform:     env.p,
		replyCtx:     "ctx",
		agent:        env.primary,
		pendingMessages: []queuedMessage{{
			messageID: "q1", platform: env.p, replyCtx: "ctx", content: "queued question",
			userID: "u1", userName: "user", msgPlatform: "test", msgSessionKey: fallbackTestKey,
		}},
	}
	if withFallbackPolicy {
		st.fallbackIntercept = interceptRetriable
	}
	env.e.interactiveMu.Lock()
	env.e.interactiveStates[fallbackTestKey] = st
	env.e.interactiveMu.Unlock()
	return st, session, gen
}

// R2-1a: /stop between popping a queued message and restarting the dead
// primary: no restart, no send, the message is dropped as stopped.
func TestAgentFallback_StopBeforeDeadPrimaryRestart(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	dead := newScriptedSession("dead-sid")
	dead.dead = true
	fresh := newScriptedSession("fresh-sid", okTurn("must not run"))
	env.primary.session = dead
	env.primary.more = []*scriptedSession{fresh}
	env.primary.starts = 1 // the dead primary was started once already
	st, session, gen := env.drainFixture(dead, true)

	// primaryBlocked runs between the pop and the restart; an expired block
	// makes it consult the clock, where the test issues /stop.
	env.e.agentFallback.mu.Lock()
	env.e.agentFallback.blockedUntil = env.now.Add(-time.Minute)
	env.e.agentFallback.mu.Unlock()
	var once sync.Once
	env.e.agentFallback.now = func() time.Time {
		once.Do(func() { env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") })
		return env.now
	}

	env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen)

	if got := env.primary.StartCount(); got != 1 {
		t.Errorf("primary starts = %d, want 1 (no restart after /stop)", got)
	}
	if got := fresh.prompts(); len(got) != 0 {
		t.Errorf("stopped queued message was sent: %q", got)
	}
	if env.sentContaining("session reset") != 1 {
		t.Errorf("stopped queued message not reported; sent = %#v", env.p.getSent())
	}
}

// R2-1b: /stop lands while the dead primary is being restarted: the restart
// may happen, but the stopped message is never sent. StartSession runs under
// interactiveMu, so the hook applies /stop's effect on the turn directly
// (state stopped, busy lock force-released) instead of calling /stop.
func TestAgentFallback_StopDuringDeadPrimaryRestart(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	dead := newScriptedSession("dead-sid")
	dead.dead = true
	fresh := newScriptedSession("fresh-sid", okTurn("must not run"))
	env.primary.session = dead
	env.primary.more = []*scriptedSession{fresh}
	st, session, gen := env.drainFixture(dead, true)
	env.primary.onStart = func() {
		st.markStopped()
		session.ForceUnlock()
	}

	env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen)

	if got := fresh.prompts(); len(got) != 0 {
		t.Errorf("stopped queued message was sent: %q", got)
	}
	if env.sentContaining("session reset") != 1 {
		t.Errorf("stopped queued message not reported; sent = %#v", env.p.getSent())
	}
}

// R2-2: /stop completes while the last fallback is inside StartTyping (after
// every earlier check passed): the send path itself refuses to send.
func TestAgentFallback_StopDuringFallbackTypingPreventsSend(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("must not run")},
	)
	var noticeSeen sync.Once
	armed := make(chan struct{})
	env.p.hookMu.Lock()
	env.p.onSend = func(c string) {
		if strings.Contains(c, "answering with") {
			noticeSeen.Do(func() { close(armed) })
		}
	}
	env.p.hookMu.Unlock()
	inTyping, release, turnEnded := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var typingOnce, endOnce sync.Once
	env.p.setTypingHooks(func() {
		select {
		case <-armed: // only the fallback's typing indicator blocks
			typingOnce.Do(func() { close(inTyping); <-release })
		default:
		}
	}, func() {
		select {
		case <-armed:
			select {
			case <-inTyping:
				endOnce.Do(func() { close(turnEnded) })
			default:
			}
		default:
		}
	})
	sent := make(chan struct{})
	env.fbs[0].session.mu.Lock()
	env.fbs[0].session.beforeTurn = func(int) { close(sent) }
	env.fbs[0].session.mu.Unlock()

	// Exactly one of these happens once the send goroutine runs: the
	// guarded send logs its cancellation, or the agent receives Send.
	cancelled := watchLog(t, "agent fallback: turn cancelled")
	env.receive("question")
	waitClosed(t, inTyping, "fallback StartTyping")
	if !env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") {
		t.Fatal("/stop found no session")
	}
	close(release)
	select {
	case <-sent:
		t.Fatal("fallback agent was sent the message after /stop")
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("send path neither sent nor cancelled")
	}
	waitClosed(t, turnEnded, "fallback turn end")
}

// R2-3: the unsolicited reader is stuck in a slow platform send past the
// stop timeout: the queued message is not sent (fail closed) and the user
// gets the "previous request still processing" notice.
func TestAgentFallback_StuckUnsolicitedReaderFailsClosed(t *testing.T) {
	old := unsolicitedReaderStopTimeout
	unsolicitedReaderStopTimeout = 50 * time.Millisecond
	t.Cleanup(func() { unsolicitedReaderStopTimeout = old })

	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	primary := newScriptedSession("primary-sid", okTurn("must not run"))
	env.primary.session = primary
	st, session, gen := env.drainFixture(primary, true)

	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	env.p.hookMu.Lock()
	env.p.onSend = func(c string) {
		if strings.Contains(c, "background result") {
			once.Do(func() { close(entered); <-release })
		}
	}
	env.p.hookMu.Unlock()
	e := env.e
	e.startUnsolicitedReader(st, session, e.sessions, fallbackTestKey, "")
	primary.events <- Event{Type: EventResult, Content: "background result", Done: true}
	waitClosed(t, entered, "unsolicited reader blocked in platform send")

	e.drainPendingMessages(st, session, e.sessions, fallbackTestKey, gen)
	close(release)

	if got := primary.prompts(); len(got) != 0 {
		t.Errorf("queued message sent while the reader still owned Events(): %q", got)
	}
	if env.sentContaining("Previous request still processing") != 1 {
		t.Errorf("no fail-closed notice; sent = %#v", env.p.getSent())
	}
}

// R2-5: hooks keep keys that merely look like fallback state keys, with or
// without a chain; only registered fallback states are mapped.
func TestAgentFallback_HookKeysUseRegistryNotSubstring(t *testing.T) {
	var mu sync.Mutex
	var got []HookEvent
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var ev HookEvent
		_ = json.NewDecoder(r.Body).Decode(&ev)
		mu.Lock()
		got = append(got, ev)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	async := false
	newHM := func() *HookManager {
		return NewHookManager("test", []HookConfig{{Event: "*", Type: "http", URL: srv.URL, Async: &async}}, "sh", "-c", "")
	}
	const external = "custom:room#fallback:0"

	// Empty chain: no mapper at all.
	plain := newFallbackEnv(t, nil)
	plain.e.SetHooks(newHM())
	plain.e.hooks.Emit(HookEvent{Event: HookEventSessionStarted, SessionKey: external})
	plain.e.hooks.Emit(HookEvent{Event: HookEventMessageSent, SessionKey: external})

	// Chain configured: an unregistered look-alike key also passes unchanged,
	// a registered fallback state key is mapped.
	chained := newFallbackEnv(t, nil, []scriptedTurn{okTurn("x")})
	chained.e.SetHooks(newHM())
	chained.e.hooks.Emit(HookEvent{Event: HookEventSessionStarted, SessionKey: external})
	chained.e.agentFallback.bindState(fallbackStateKey("chat", 0), "chat", nil, chained.e.agentFallback.registerState(fallbackStateKey("chat", 0), "chat"))
	chained.e.hooks.Emit(HookEvent{Event: HookEventMessageSent, SessionKey: fallbackStateKey("chat", 0)})
	chained.e.hooks.Emit(HookEvent{Event: HookEventSessionEnded, SessionKey: fallbackStateKey("chat", 0)})

	mu.Lock()
	defer mu.Unlock()
	want := []struct {
		ev  HookEventType
		key string
	}{
		{HookEventSessionStarted, external},
		{HookEventMessageSent, external},
		{HookEventSessionStarted, external},
		{HookEventMessageSent, "chat"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d hook calls, want %d: %+v", len(got), len(want), got)
	}
	for i, w := range want {
		if got[i].Event != w.ev || got[i].SessionKey != w.key {
			t.Errorf("hook %d = (%s, %q), want (%s, %q)", i, got[i].Event, got[i].SessionKey, w.ev, w.key)
		}
	}
}

// R2-6: with no fallback chain, draining a queued message behaves as on
// main: no unsolicited reader is started and no idle close is armed by the
// drain loop.
func TestAgentFallback_EmptyChainDrainKeepsLifecycle(t *testing.T) {
	env := newFallbackEnv(t, nil)
	env.e.SetAgentSessionIdleTimeout(time.Hour)
	primary := newScriptedSession("primary-sid", okTurn("queued answer"))
	env.primary.session = primary
	st, session, gen := env.drainFixture(primary, false)

	if !env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen) {
		t.Fatal("drain did not unlock the session")
	}
	env.waitSent("queued answer")
	st.mu.Lock()
	reader, idle := st.unsolicitedCancel != nil, st.agentSessionIdleCancel != nil
	st.mu.Unlock()
	if reader {
		t.Error("drain started an unsolicited reader without a fallback chain")
	}
	if idle {
		t.Error("drain armed the idle close without a fallback chain")
	}
}

// ---------- review round 3 ----------

// setDispatchHook installs testHookGuardedDispatch for one test: it signals
// entered and blocks until release is closed (first dispatch only).
func setDispatchHook(t *testing.T) (entered, release chan struct{}) {
	t.Helper()
	entered, release = make(chan struct{}), make(chan struct{})
	var once sync.Once
	testHookGuardedDispatch = func() {
		once.Do(func() {
			close(entered)
			<-release
		})
	}
	t.Cleanup(func() { testHookGuardedDispatch = nil })
	return entered, release
}

func (env *fallbackEnv) primaryState() *interactiveState {
	env.e.interactiveMu.Lock()
	defer env.e.interactiveMu.Unlock()
	return env.e.interactiveStates[fallbackTestKey]
}

func waitTrue(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// R3-1a: /stop starts while a fallback send sits between its ownership
// check and Send: the send is refused and nothing reaches the agent; /stop
// completes after the dispatch let go.
func TestAgentFallback_StopBetweenCheckAndSendFallback(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("must not run")},
	)
	entered, release := setDispatchHook(t)
	cancelled := watchLog(t, "agent fallback: turn cancelled")

	env.receive("question")
	waitClosed(t, entered, "guarded dispatch")
	st := env.primaryState()
	stopDone := make(chan bool, 1)
	go func() { stopDone <- env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") }()
	waitTrue(t, "/stop to request the stop", st.stopRequested.Load)
	select {
	case <-stopDone:
		t.Fatal("/stop completed while a dispatch held the gate")
	default:
	}
	close(release)
	waitClosed(t, cancelled, "cancelled dispatch")
	select {
	case ok := <-stopDone:
		if !ok {
			t.Fatal("/stop found no session")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("/stop did not complete")
	}
	if got := env.fbs[0].session.prompts(); len(got) != 0 {
		t.Errorf("stopped prompt reached the fallback agent: %q", got)
	}
}

// R3-1b: same window on the drain path after a dead-primary restart.
func TestAgentFallback_StopBetweenCheckAndSendRestartedPrimary(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	dead := newScriptedSession("dead-sid")
	dead.dead = true
	fresh := newScriptedSession("fresh-sid", okTurn("must not run"))
	env.primary.session = dead
	env.primary.more = []*scriptedSession{fresh}
	env.primary.starts = 1 // the dead primary was started once already
	st, session, gen := env.drainFixture(dead, true)
	entered, release := setDispatchHook(t)

	drained := make(chan struct{})
	go func() {
		env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen)
		close(drained)
	}()
	waitClosed(t, entered, "guarded dispatch")
	restarted := env.primaryState()
	if restarted == st {
		t.Fatal("primary was not restarted")
	}
	stopDone := make(chan bool, 1)
	go func() { stopDone <- env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") }()
	waitTrue(t, "/stop to request the stop", restarted.stopRequested.Load)
	close(release)
	waitClosed(t, drained, "drain")
	<-stopDone
	if got := fresh.prompts(); len(got) != 0 {
		t.Errorf("stopped prompt reached the restarted primary: %q", got)
	}
}

// R3-2: after a nested drain released generation G and a newer turn holds
// G+1 with C queued, the old drain leaves C alone; C is processed exactly
// once, by the owner, and never reported as "session reset".
func TestAgentFallback_StaleDrainLeavesNewerQueue(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	primary := newScriptedSession("primary-sid", okTurn("answer to C"))
	env.primary.session = primary
	st, session, oldGen := env.drainFixture(primary, true)
	session.Unlock(oldGen) // nested drain (auto-compress) released G
	newGen, ok := session.TryLock()
	if !ok {
		t.Fatal("turn B could not take the lock")
	}

	if env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, oldGen) {
		t.Fatal("stale drain claims it unlocked the session")
	}
	st.mu.Lock()
	queued := len(st.pendingMessages)
	st.mu.Unlock()
	if queued != 1 {
		t.Fatalf("stale drain touched the newer queue: %d left, want 1", queued)
	}
	if len(primary.prompts()) != 0 || env.sentContaining("session reset") != 0 {
		t.Fatalf("stale drain processed or dropped C; sent = %#v", env.p.getSent())
	}

	// B's drain owns the queue and processes C once.
	if !env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, newGen) {
		t.Fatal("owner drain did not unlock")
	}
	env.waitSent("answer to C")
	if got := len(primary.prompts()); got != 1 {
		t.Errorf("C sent %d times, want 1", got)
	}
	if env.sentContaining("session reset") != 0 {
		t.Errorf("C reported as session reset: %#v", env.p.getSent())
	}
}

// R3-3a: the primary died on a retriable error and a fallback answers for
// longer than the stale-lock threshold: a new message must not break the
// lock and start a second turn.
func TestAgentFallback_StaleLockKeepsLiveFallbackTurn(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{errorTurn(`API Error: {"error":{"type":"rate_limit_error"}}`, ErrorKindRateLimit)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	env.e.SetStaleLockBreakAfter(time.Nanosecond) // every held lock looks stale
	env.primary.session.dieAfter = map[int]bool{0: true}
	entered, release := gate(env.fbs[0].session, 0)

	env.receive("first")
	waitClosed(t, entered, "fallback turn")
	env.receive("second")
	if got := env.sentContaining("Previous request still processing"); got != 1 {
		t.Errorf("second message not rejected as still processing; sent = %#v", env.p.getSent())
	}
	if got := env.sentContaining("会话卡锁已自动恢复"); got != 0 {
		t.Errorf("stale-lock recovery broke a live fallback turn's lock")
	}
	close(release)
	env.waitSent("fallback answer")
	env.waitIdle()
	if got := env.primary.StartCount(); got != 1 {
		t.Errorf("primary started %d times, want 1", got)
	}
	if got := len(env.fbs[0].session.prompts()); got != 1 {
		t.Errorf("fallback got %d prompts, want 1", got)
	}
}

// R3-3b: a second fallback run for the same conversation while the first
// is in flight is refused by the fallback session's busy lock, so the
// fallback state never gets two senders or two readers.
func TestAgentFallback_FallbackStateBusyLockRejectsSecondTurn(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	entered, release := gate(env.fbs[0].session, 0)
	env.receive("first")
	waitClosed(t, entered, "fallback turn")

	st := env.primaryState()
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	session.mu.Lock()
	gen := session.lockGen
	session.mu.Unlock()
	second := make(chan struct{})
	go func() {
		guard := fallbackGuard{primary: st, session: session, sessions: env.e.sessions, lockGen: gen, interactiveKey: fallbackTestKey}
		env.e.answerWithFallbacks(guard, turnOutcome{Intercepted: true, Kind: ErrorKindRateLimit, Input: queuedMessage{
			messageID: "m2", platform: env.p, replyCtx: "ctx", content: "second", msgSessionKey: fallbackTestKey,
		}})
		close(second)
	}()
	select {
	case <-second:
	case <-time.After(3 * time.Second):
		t.Fatal("second fallback run did not return (it shared the busy fallback state)")
	}
	if got := len(env.fbs[0].session.prompts()); got != 1 {
		t.Errorf("fallback got %d prompts, want 1", got)
	}
	if got := env.sentContaining("Previous request still processing"); got != 1 {
		t.Errorf("second run not rejected; sent = %#v", env.p.getSent())
	}
	close(release)
	env.waitSent("fallback answer")
	env.waitIdle()
}

// R3-4: registry entries go away with their state, so a real conversation
// whose key happens to equal an old fallback state key is left alone.
func TestAgentFallback_RegistryForgetsRemovedStates(t *testing.T) {
	var mu sync.Mutex
	var got []HookEvent
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var ev HookEvent
		_ = json.NewDecoder(r.Body).Decode(&ev)
		mu.Lock()
		got = append(got, ev)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("fallback one")},
	)
	// cleanupInteractiveState closes both sessions; the second message gets
	// fresh ones.
	env.primary.more = []*scriptedSession{newScriptedSession("primary-sid-2", usageLimitTurn(time.Time{}))}
	env.fbs[0].more = []*scriptedSession{newScriptedSession("fb0-sid-2", okTurn("fallback two"))}
	async := false
	env.e.SetHooks(NewHookManager("test", []HookConfig{{Event: "message.sent", Type: "http", URL: srv.URL, Async: &async}}, "sh", "-c", ""))
	fbKey := fallbackStateKey(fallbackTestKey, 0)

	// Cleanup path.
	env.send("one")
	env.waitSent("fallback one")
	if _, ok := env.e.agentFallback.ownerOf(fbKey); !ok {
		t.Fatal("fallback state not registered after its turn")
	}
	env.e.cleanupInteractiveState(fallbackTestKey)
	if _, ok := env.e.agentFallback.ownerOf(fbKey); ok {
		t.Error("registry kept a cleaned-up fallback state")
	}

	// /stop path.
	env.send("two")
	env.waitSent("fallback two")
	env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx")
	if _, ok := env.e.agentFallback.ownerOf(fbKey); ok {
		t.Error("registry kept a stopped fallback state")
	}
	if keys := env.e.agentFallback.ownedStatesOf(fallbackTestKey); len(keys) != 0 {
		t.Errorf("cleanup/permission lookup would still target %v", keys)
	}

	// A real conversation using that literal key passes through unchanged.
	mu.Lock()
	got = nil
	mu.Unlock()
	env.e.hooks.Emit(HookEvent{Event: HookEventMessageSent, SessionKey: fbKey})
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || got[0].SessionKey != fbKey {
		t.Errorf("real key rewritten: %+v", got)
	}
}

// R3-5: without a chain, the drain calls the agent's Alive() exactly once
// before Send, as on main.
func TestAgentFallback_EmptyChainDrainAliveCalls(t *testing.T) {
	env := newFallbackEnv(t, nil)
	primary := newScriptedSession("primary-sid", okTurn("queued answer"))
	env.primary.session = primary
	atSend := make(chan int32, 1)
	primary.beforeTurn = func(int) { atSend <- primary.aliveCalls.Load() }
	st, session, gen := env.drainFixture(primary, false)
	env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen)
	if got := <-atSend; got != 1 {
		t.Errorf("Alive() called %d times before Send, want 1 (main)", got)
	}
}

// ---------- review round 4 ----------

// R4-1: the state registered for the key is replaced between /stop's
// capture and its gate acquisition, while the replacement's dispatch is
// already inside Send. /stop must retarget the replacement: it raises the
// replacement's stop request, waits for its gate, and only then stops it;
// a later dispatch on it is refused.
func TestAgentFallback_StopRetargetsReplacedState(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	gen, _ := session.TryLock()
	a := &interactiveState{agentSession: newScriptedSession("a"), busySession: session, platform: env.p, replyCtx: "ctx", fallbackIntercept: interceptRetriable}
	b := &interactiveState{agentSession: newScriptedSession("b"), busySession: session, platform: env.p, replyCtx: "ctx", fallbackIntercept: interceptRetriable}
	env.e.interactiveMu.Lock()
	env.e.interactiveStates[fallbackTestKey] = a
	env.e.interactiveMu.Unlock()

	live := func() bool { return !b.isStopped() && session.HoldsLock(gen) }
	inSend, releaseSend, sendReturned := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		_ = guardedDispatch(b, live, func() error {
			close(inSend)
			<-releaseSend
			return nil
		})
		close(sendReturned)
	}()
	waitClosed(t, inSend, "replacement dispatch inside Send")

	var once sync.Once
	testHookStopAfterCapture = func() {
		once.Do(func() { // dead-primary restart replaces A with B
			env.e.interactiveMu.Lock()
			env.e.interactiveStates[fallbackTestKey] = b
			env.e.interactiveMu.Unlock()
		})
	}
	t.Cleanup(func() { testHookStopAfterCapture = nil })

	stopDone := make(chan bool, 1)
	go func() { stopDone <- env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") }()
	waitTrue(t, "/stop to retarget the replacement", b.stopRequested.Load)
	select {
	case <-stopDone:
		t.Fatal("/stop completed while the replacement's dispatch held its gate")
	default:
	}
	if b.isStopped() {
		t.Fatal("replacement marked stopped while its Send was in flight")
	}
	close(releaseSend)
	waitClosed(t, sendReturned, "replacement Send")
	if !<-stopDone {
		t.Fatal("/stop found no session")
	}
	if !b.isStopped() || a.isStopped() {
		t.Errorf("stopped a=%v b=%v, want only the replacement", a.isStopped(), b.isStopped())
	}
	if err := guardedDispatch(b, live, func() error { t.Error("dispatch after /stop was sent"); return nil }); !errors.Is(err, errTurnCancelled) {
		t.Errorf("dispatch after /stop = %v, want errTurnCancelled", err)
	}
}

// R4-2: the primary failed and the fallback notice is still being sent when
// a second message arrives with an expired stale-lock threshold: the
// conversation is in hand-off, so the lock is not broken and the original
// turn still gets its fallback answer.
func TestAgentFallback_StaleLockKeepsTurnDuringHandoff(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{errorTurn(`API Error: {"error":{"type":"rate_limit_error"}}`, ErrorKindRateLimit)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	env.e.SetStaleLockBreakAfter(time.Nanosecond)
	env.primary.session.dieAfter = map[int]bool{0: true}
	inNotice, releaseNotice := make(chan struct{}), make(chan struct{})
	var once sync.Once
	env.p.hookMu.Lock()
	env.p.onSend = func(c string) {
		if strings.Contains(c, "answering with") {
			once.Do(func() { close(inNotice); <-releaseNotice })
		}
	}
	env.p.hookMu.Unlock()

	env.receive("first")
	waitClosed(t, inNotice, "fallback notice")
	env.receive("second")
	if got := env.sentContaining("Previous request still processing"); got != 1 {
		t.Errorf("second message not rejected during hand-off; sent = %#v", env.p.getSent())
	}
	if env.sentContaining("会话卡锁已自动恢复") != 0 {
		t.Error("stale-lock recovery broke the lock during the hand-off")
	}
	close(releaseNotice)
	env.waitSent("fallback answer")
	env.waitIdle()
	if env.e.agentFallback.inHandoff(fallbackTestKey) {
		t.Error("hand-off not ended after the chain finished")
	}
}

// R4-3: an old cleanup of fallback state A must not unregister B, which a
// newer turn registered under the same key right after A was removed; /stop
// still finds and stops B.
func TestAgentFallback_CleanupKeepsReplacementRegistration(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	primary := &interactiveState{agentSession: newScriptedSession("p"), busySession: session, platform: env.p, replyCtx: "ctx"}
	fbKey := fallbackStateKey(fallbackTestKey, 0)
	a := &interactiveState{agentSession: newScriptedSession("fa"), busySession: NewSessionManager("").GetOrCreateActive("x"), platform: env.p, replyCtx: "ctx"}
	b := &interactiveState{agentSession: newScriptedSession("fb"), busySession: NewSessionManager("").GetOrCreateActive("y"), platform: env.p, replyCtx: "ctx"}
	chain := env.e.agentFallback
	env.e.interactiveMu.Lock()
	env.e.interactiveStates[fallbackTestKey] = primary
	env.e.interactiveStates[fbKey] = a
	env.e.interactiveMu.Unlock()
	chain.bindState(fbKey, fallbackTestKey, a, chain.registerState(fbKey, fallbackTestKey))

	var once sync.Once
	testHookAfterStateRemoved = func(key string) {
		if key != fbKey {
			return
		}
		once.Do(func() { // a new turn registers B under the same key
			claim := chain.registerState(fbKey, fallbackTestKey)
			env.e.interactiveMu.Lock()
			env.e.interactiveStates[fbKey] = b
			env.e.interactiveMu.Unlock()
			chain.bindState(fbKey, fallbackTestKey, b, claim)
		})
	}
	t.Cleanup(func() { testHookAfterStateRemoved = nil })

	env.e.cleanupInteractiveState(fbKey, a)
	if _, ok := chain.ownerOf(fbKey); !ok {
		t.Fatal("old cleanup unregistered the replacement fallback state")
	}
	env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx")
	if !b.isStopped() {
		t.Error("/stop did not stop the replacement fallback state")
	}
}

// R4-4a: a terminal error from a fallback (max turns) is shown; the chain
// does not advance to the next fallback.
func TestAgentFallback_TerminalErrorDoesNotAdvance(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{errorTurn("Reached max turns (25)", ErrorKindMaxTurns)},
		[]scriptedTurn{okTurn("must not run")},
	)
	env.send("question")
	env.waitSent("Reached max turns (25)")
	if n := env.builtCount(1); n != 0 {
		t.Errorf("fallback[1] built %d times after a terminal error, want 0", n)
	}
	if got := env.sentContaining("answering with OpenCode (model-1)"); got != 0 {
		t.Errorf("chain advanced after a terminal error; sent = %#v", env.p.getSent())
	}
}

// R4-4b: an auth/provider error from a fallback advances to the next one.
func TestAgentFallback_ProviderErrorAdvances(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{errorTurn("ProviderAuthError: invalid api key", ErrorKindUnknown)},
		[]scriptedTurn{okTurn("second fallback answer")},
	)
	env.send("question")
	env.waitSent("second fallback answer")
	if env.sentContaining("invalid api key") != 0 {
		t.Errorf("provider error shown instead of advancing: %#v", env.p.getSent())
	}
}

func TestErrorKindIsTerminal(t *testing.T) {
	for _, k := range []ErrorKind{ErrorKindMaxTurns, ErrorKindMaxBudget, ErrorKindCancelled, ErrorKindContextTooLong} {
		if !k.IsTerminal() || k.IsRetriable() {
			t.Errorf("%q: terminal=%v retriable=%v", k, k.IsTerminal(), k.IsRetriable())
		}
	}
	for _, k := range []ErrorKind{ErrorKindUnknown, ErrorKindRateLimit, ErrorKindOverloaded, ErrorKindUsageLimit} {
		if k.IsTerminal() {
			t.Errorf("%q must not be terminal", k)
		}
	}
}

// ---------- review round 5 ----------

// R5-1: the next turn claims the fallback key before it waits for the old
// state A to be torn down; A's cleanup then removes A but not the claim,
// and the new state B is bound to it: B is registered and /stop stops it.
func TestAgentFallback_ClaimSurvivesOldStateCleanup(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	session := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	primary := &interactiveState{agentSession: newScriptedSession("p"), busySession: session, platform: env.p, replyCtx: "ctx"}
	fbKey := fallbackStateKey(fallbackTestKey, 0)
	a := &interactiveState{agentSession: newScriptedSession("fa"), busySession: NewSessionManager("").GetOrCreateActive("x"), platform: env.p, replyCtx: "ctx"}
	b := &interactiveState{agentSession: newScriptedSession("fb"), busySession: NewSessionManager("").GetOrCreateActive("y"), platform: env.p, replyCtx: "ctx"}
	chain := env.e.agentFallback
	env.e.interactiveMu.Lock()
	env.e.interactiveStates[fallbackTestKey] = primary
	env.e.interactiveStates[fbKey] = a
	env.e.interactiveMu.Unlock()
	chain.bindState(fbKey, fallbackTestKey, a, chain.registerState(fbKey, fallbackTestKey))

	// The next turn claims the key (runFallbackTurn), then waits for A.
	claim := chain.registerState(fbKey, fallbackTestKey)
	var once sync.Once
	testHookAfterStateRemoved = func(key string) {
		if key != fbKey {
			return
		}
		once.Do(func() {
			// Between A's removal and B's creation the claim must hold.
			if keys := chain.ownedStatesOf(fallbackTestKey); len(keys) != 1 {
				t.Errorf("claimed key not registered after A's removal: %v", keys)
			}
			env.e.interactiveMu.Lock()
			env.e.interactiveStates[fbKey] = b
			env.e.interactiveMu.Unlock()
			chain.bindState(fbKey, fallbackTestKey, b, claim)
		})
	}
	t.Cleanup(func() { testHookAfterStateRemoved = nil })

	env.e.cleanupInteractiveState(fbKey, a)
	chain.mu.Lock()
	reg, ok := chain.owned[fbKey]
	chain.mu.Unlock()
	if !ok || reg.state != b || reg.claim != 0 {
		t.Fatalf("registration = %+v (ok=%v), want bound to B with no open claim", reg, ok)
	}
	env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx")
	if !b.isStopped() {
		t.Error("/stop did not stop B")
	}
}

// R5-2a: during a reset window, the primary's idle close is armed (its
// last turn ended cleanly with an empty queue) when a message is queued and
// takes the blocked-primary shortcut. The shortcut must disarm the close
// while the fallback answers, or the close tears the conversation and the
// running fallback down.
func TestAgentFallback_QueuedBlockedTurnDisarmsPrimaryIdleClose(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("fb answer")})
	env.e.SetAgentSessionIdleTimeout(30 * time.Millisecond)
	env.e.agentFallback.recordPrimaryFailure(ErrorKindUsageLimit, env.now.Add(time.Hour))
	primary := newScriptedSession("primary-sid")
	env.primary.session = primary
	st, session, gen := env.drainFixture(primary, true)
	// Arm the idle close as the end of the previous turn did, before the
	// message was queued (scheduling refuses while the queue is non-empty).
	st.mu.Lock()
	queued := st.pendingMessages
	st.pendingMessages = nil
	st.mu.Unlock()
	env.e.scheduleAgentSessionIdleClose(fallbackTestKey, st)
	st.mu.Lock()
	armed := st.agentSessionIdleCancel != nil
	st.pendingMessages = queued
	st.mu.Unlock()
	if !armed {
		t.Fatal("precondition: idle close not armed")
	}
	entered, release := gate(env.fbs[0].session, 0)

	drained := make(chan struct{})
	go func() {
		env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen)
		close(drained)
	}()
	waitClosed(t, entered, "queued fallback turn")
	time.Sleep(150 * time.Millisecond) // an armed 30 ms close would fire now
	close(release)
	env.waitSent("fb answer")
	waitClosed(t, drained, "drain")
	if env.primaryState() != st {
		t.Error("primary state was closed by its idle timer during the fallback turn")
	}
}

// R5-2b: /stop stops a running fallback even when the primary state is
// already gone (e.g. removed by its idle close).
func TestAgentFallback_StopWithoutPrimaryStopsFallback(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	entered, release := gate(env.fbs[0].session, 0)
	env.receive("question")
	waitClosed(t, entered, "fallback turn")
	env.e.interactiveMu.Lock()
	fbState := env.e.interactiveStates[fallbackStateKey(fallbackTestKey, 0)]
	delete(env.e.interactiveStates, fallbackTestKey) // primary gone
	env.e.interactiveMu.Unlock()

	if !env.e.stopInteractiveSession(fallbackTestKey, env.p, "ctx") {
		t.Error("/stop reported nothing to stop while a fallback was running")
	}
	if fbState == nil || !fbState.isStopped() {
		t.Error("fallback state not stopped")
	}
	close(release)
}

// R5-3: the hand-off is recorded before any blocking work of the
// interception (here: the primary's Send still returning). A second message
// with an expired stale-lock threshold is rejected, and the original turn
// gets its fallback answer.
func TestAgentFallback_HandoffRecordedBeforeBlockingWork(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{errorTurn(`API Error: {"error":{"type":"rate_limit_error"}}`, ErrorKindRateLimit)},
		[]scriptedTurn{okTurn("fallback answer")},
	)
	env.e.SetStaleLockBreakAfter(time.Nanosecond)
	primary := env.primary.session
	primary.dieAfter = map[int]bool{0: true}
	inSend, releaseSend := make(chan struct{}), make(chan struct{})
	primary.afterTurn = func(int) {
		close(inSend)
		<-releaseSend // Send has emitted the error but not returned yet
	}

	env.receive("first")
	waitClosed(t, inSend, "primary Send")
	// Give the event loop time to reach the interception, which then waits
	// for the blocked Send.
	deadline := time.Now().Add(time.Second)
	for !env.e.agentFallback.inHandoff(fallbackTestKey) && time.Now().Before(deadline) {
		time.Sleep(2 * time.Millisecond)
	}
	env.receive("second")
	if got := env.sentContaining("Previous request still processing"); got != 1 {
		t.Errorf("second message not rejected; sent = %#v", env.p.getSent())
	}
	if env.sentContaining("会话卡锁已自动恢复") != 0 {
		t.Error("stale-lock recovery broke the lock before the hand-off was recorded")
	}
	close(releaseSend)
	env.waitSent("fallback answer")
	env.waitIdle()
}

// R5-5: engine Stop answers a fallback's pending permission (deny) so the
// waiting turn returns.
func TestAgentFallback_StopResolvesPendingPermission(t *testing.T) {
	env := newFallbackEnv(t,
		[]scriptedTurn{usageLimitTurn(time.Time{})},
		[]scriptedTurn{{{Type: EventPermissionRequest, RequestID: "p1", ToolName: "Bash", ToolInput: "rm -rf x"}}},
	)
	env.receive("question")
	fbKey := fallbackStateKey(fallbackTestKey, 0)
	waitTrue(t, "fallback permission request", func() bool {
		_, pending := env.e.lookupPending(fbKey)
		return pending != nil
	})
	resolved := watchLog(t, "permission resolved")
	if err := env.e.Stop(); err != nil {
		t.Logf("Stop: %v", err)
	}
	waitClosed(t, resolved, "pending permission to resolve on Stop")
	perms := env.fbs[0].session.permissions()
	if len(perms) != 1 || perms[0].Behavior != "deny" {
		t.Errorf("permission answers = %+v, want one deny", perms)
	}
	if env.e.agentFallback.inHandoff(fallbackTestKey) {
		t.Error("hand-off survived engine Stop")
	}
}
