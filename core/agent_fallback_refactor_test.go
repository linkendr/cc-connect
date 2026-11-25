package core

// Regression tests for the fork's engine-hook refactor (PR 10 review):
// a handed-back queued turn acts on that turn's own preview, and a turn
// cancelled by /stop still ends its fallback hand-off after the state is
// reused by a new turn.

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type queuedPreviewPlatform struct {
	mockUpdaterPlatform
	deleted []any
}

func (p *queuedPreviewPlatform) SendPreviewStart(_ context.Context, rc any, _ string) (any, error) {
	return rc, nil
}

func (p *queuedPreviewPlatform) DeletePreviewMessage(_ context.Context, handle any) error {
	p.deleted = append(p.deleted, handle)
	return nil
}

// The primary answers the first message, picks up a queued one in the same
// event loop, streams part of an answer, then hits a usage limit.
func TestFallbackInterceptDiscardsQueuedTurnPreview(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("unused")})
	p := &queuedPreviewPlatform{}
	p.n = "test"
	env.e.streamPreview = StreamPreviewCfg{Enabled: true, MinDeltaChars: 1, IntervalMs: 1}
	as := newScriptedSession("primary", scriptedTurn{
		{Type: EventText, Content: "partial queued answer"},
		{Type: EventError, ErrorKind: ErrorKindUsageLimit},
	})
	s := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	gen, _ := s.TryLock()
	defer s.Unlock(gen)
	st := &interactiveState{agentSession: as, platform: p, replyCtx: "first", agent: env.e.agent}
	st.fallbackIntercept = interceptRetriable
	st.pendingMessages = []queuedMessage{{messageID: "second", content: "second question", platform: p, replyCtx: "second"}}
	as.events <- Event{Type: EventResult, Content: "first answer", Done: true}
	env.e.processInteractiveEvents(st, s, env.e.sessions, fallbackTestKey, "first", time.Now(), nil, nil, "first", gen)
	for _, h := range p.deleted {
		if h == "second" {
			return
		}
	}
	t.Fatalf("queued fallback left its partial preview visible: deleted handles = %v, want second", p.deleted)
}

type cancelTurnSession struct{ *scriptedSession }

func (s *cancelTurnSession) CancelTurn() error { return nil }

type cancelTurnAgent struct{ *scriptedAgent }

func (a *cancelTurnAgent) StartSession(ctx context.Context, id string) (AgentSession, error) {
	as, err := a.scriptedAgent.StartSession(ctx, id)
	if err != nil {
		return nil, err
	}
	return &cancelTurnSession{as.(*scriptedSession)}, nil
}

// The first turn hits a usage limit and blocks in its typing-stop defer;
// /stop cancels it (state kept), a second turn completes on the same
// state, then the first caller finishes.
func TestFallbackCancelledTurnOutcomeEndsHandoff(t *testing.T) {
	env := newFallbackEnv(t, []scriptedTurn{usageLimitTurn(time.Time{}), okTurn("new answer")}, []scriptedTurn{okTurn("unused")})
	env.e.agent = &cancelTurnAgent{env.primary}
	paused, release, oldDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	env.p.setTypingHooks(nil, func() {
		if calls.Add(1) == 1 {
			close(paused)
			<-release
		}
	})
	s := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	gen, _ := s.TryLock()
	msg := func(id string) *Message {
		return &Message{SessionKey: fallbackTestKey, Platform: "test", MessageID: id, Content: id, ReplyCtx: "ctx"}
	}
	go func() {
		defer close(oldDone)
		env.e.processInteractiveMessage(env.p, msg("first"), s, gen)
	}()
	select {
	case <-paused:
	case <-time.After(3 * time.Second):
		t.Fatal("first event loop did not reach its typing-stop defer")
	}
	if !env.e.stopInteractiveSessionSilently(fallbackTestKey) {
		t.Fatal("stop did not cancel the first turn")
	}
	newGen, ok := s.TryLock()
	if !ok {
		t.Fatal("cancel did not release the busy lock")
	}
	env.e.processInteractiveMessage(env.p, msg("second"), s, newGen)
	unblock()
	select {
	case <-oldDone:
	case <-time.After(3 * time.Second):
		t.Fatal("old caller did not finish")
	}
	if env.e.agentFallback.inHandoff(fallbackTestKey) {
		t.Fatal("cancelled first turn lost its outcome: handoff remains live after both callers finished")
	}
}

// envRecordingAgent records the session env the engine injects.
type envRecordingAgent struct {
	*scriptedAgent
	mu  sync.Mutex
	env []string
}

func (a *envRecordingAgent) SetSessionEnv(env []string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.env = append([]string(nil), env...)
}

func (a *envRecordingAgent) sessionKeyEnv() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, kv := range a.env {
		if v, ok := strings.CutPrefix(kv, "CC_SESSION_KEY="); ok {
			return v
		}
	}
	return ""
}

func recordFallbackEnv(env *fallbackEnv) *envRecordingAgent {
	rec := &envRecordingAgent{scriptedAgent: env.fbs[0]}
	env.e.agentFallback.factory = func(string, map[string]any) (Agent, error) {
		return rec, nil
	}
	return rec
}

// The event loop answers a handed-back primary turn itself. The fallback
// agent still gets the caller's CC_SESSION_KEY (a cron turn's run key),
// not the interactive key.
func TestFallbackHandBackKeepsCallerSessionKey(t *testing.T) {
	env := newFallbackEnv(t, []scriptedTurn{usageLimitTurn(time.Time{})}, []scriptedTurn{okTurn("fallback answer")})
	rec := recordFallbackEnv(env)
	s := env.e.sessions.GetOrCreateActive(fallbackTestKey)
	gen, _ := s.TryLock()
	msg := &Message{SessionKey: "run-key", Platform: "test", MessageID: "m1", Content: "question", ReplyCtx: "ctx"}
	env.e.processInteractiveMessageWith(env.p, msg, s, env.e.agent, env.e.sessions, fallbackTestKey, "", "run-key", gen)
	if env.sentContaining("fallback answer") == 0 {
		t.Fatalf("handed-back turn not answered by the fallback; sent = %#v", env.p.getSent())
	}
	if got := rec.sessionKeyEnv(); got != "run-key" {
		t.Errorf("fallback CC_SESSION_KEY = %q, want the caller's run-key", got)
	}
}

// A queued turn the drain hands back uses the interactive key, as before,
// even when an earlier turn on the state set another caller key.
func TestFallbackDrainHandBackUsesInteractiveKey(t *testing.T) {
	env := newFallbackEnv(t, nil, []scriptedTurn{okTurn("fallback answer")})
	rec := recordFallbackEnv(env)
	st, session, gen := env.drainFixture(newScriptedSession("primary-sid", usageLimitTurn(time.Time{})), true)
	st.turnCCKey = "earlier-run-key"
	env.e.drainPendingMessages(st, session, env.e.sessions, fallbackTestKey, gen)
	if env.sentContaining("fallback answer") == 0 {
		t.Fatalf("handed-back queued turn not answered by the fallback; sent = %#v", env.p.getSent())
	}
	if got := rec.sessionKeyEnv(); got != fallbackTestKey {
		t.Errorf("fallback CC_SESSION_KEY = %q, want %q", got, fallbackTestKey)
	}
}
