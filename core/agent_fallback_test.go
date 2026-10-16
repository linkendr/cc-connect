package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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
	dieAfter   map[int]bool       // mark dead after emitting turn n
	closeAfter map[int]bool       // close the event channel after emitting turn n
	onAllow    scriptedTurn       // emitted when a permission is allowed
	perms      []PermissionResult // RespondPermission calls
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
	mu      sync.Mutex
	starts  int
}

func (a *scriptedAgent) Name() string        { return a.name }
func (a *scriptedAgent) DisplayName() string { return a.display }
func (a *scriptedAgent) StartSession(_ context.Context, _ string) (AgentSession, error) {
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
	hookMu sync.Mutex
	onSend func(content string)
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

	env.receive("question")
	waitClosed(t, stopped, "/stop after notice")
	// Negative check: give the turn goroutine time to (wrongly) send.
	select {
	case <-sent:
		t.Fatal("fallback agent was sent the stopped message")
	case <-time.After(300 * time.Millisecond):
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

func TestExternalSessionKey(t *testing.T) {
	cases := map[string]string{
		"discord:chan:thread":                    "discord:chan:thread",
		"discord:chan:thread#fallback:0":         "discord:chan:thread",
		"tg:1:1#cron:abc#fallback:2":             "tg:1:1#cron:abc",
		fallbackStateKey("ws:/repo:slack:C1", 1): "ws:/repo:slack:C1",
		"":                                       "",
	}
	for in, want := range cases {
		if got := externalSessionKey(in); got != want {
			t.Errorf("externalSessionKey(%q) = %q, want %q", in, got, want)
		}
	}
}
