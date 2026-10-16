package core

import (
	"context"
	"errors"
	"fmt"
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
}

func newScriptedSession(id string, turns ...scriptedTurn) *scriptedSession {
	return &scriptedSession{queuingAgentSession: newQueuingSession(id), turns: turns}
}

func (s *scriptedSession) Send(prompt string, messageID string, images []ImageAttachment, files []FileAttachment) error {
	_ = s.queuingAgentSession.Send(prompt, messageID, images, files)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.sends >= len(s.turns) {
		s.sends++
		return nil
	}
	turn := s.turns[s.sends]
	s.sends++
	for _, ev := range turn {
		s.events <- ev // buffered (16); scripts are short
	}
	return nil
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
	return !s.closed
}

func (s *scriptedSession) prompts() []string {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	return append([]string(nil), s.sendCalls...)
}

// scriptedAgent hands out one scripted session (reused for every turn).
type scriptedAgent struct {
	name    string
	display string
	session *scriptedSession
	mu      sync.Mutex
	starts  int
}

func (a *scriptedAgent) Name() string        { return a.name }
func (a *scriptedAgent) DisplayName() string { return a.display }
func (a *scriptedAgent) StartSession(_ context.Context, _ string) (AgentSession, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.starts++
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

type fallbackEnv struct {
	t        *testing.T
	e        *Engine
	p        *stubPlatformEngine
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
		p:       &stubPlatformEngine{n: "test"},
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
