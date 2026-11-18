package core

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"
)

// promptHookSession is a controllableAgentSession that reports every prompt.
type promptHookSession struct {
	*controllableAgentSession
	onSend func(prompt string)
}

func (s *promptHookSession) Send(prompt string, _ string, _ []ImageAttachment, _ []FileAttachment) error {
	if s.onSend != nil {
		s.onSend(prompt)
	}
	return nil
}

// gatedFirstReplyPlatform blocks the first outgoing message whose text is
// gateText until release is closed. The turn that sends it stays inside a
// platform call, like a consumer waiting on a slow chat API.
type gatedFirstReplyPlatform struct {
	stubPlatformEngine
	gateText string
	entered  chan struct{}
	release  chan struct{}
	once     sync.Once
}

func (p *gatedFirstReplyPlatform) gate(content string) {
	if content != p.gateText {
		return
	}
	first := false
	p.once.Do(func() { first = true })
	if first {
		close(p.entered)
		<-p.release
	}
}

func (p *gatedFirstReplyPlatform) Reply(ctx context.Context, rc any, content string) error {
	p.gate(content)
	return p.stubPlatformEngine.Reply(ctx, rc, content)
}

func (p *gatedFirstReplyPlatform) Send(ctx context.Context, rc any, content string) error {
	p.gate(content)
	return p.stubPlatformEngine.Send(ctx, rc, content)
}

// TestStopInteractiveSession_LateCancelResultDoesNotFinishNextTurn covers the
// graceful /stop handoff. Turn A's consumer is inside a platform call when
// /stop runs. CancelTurn returns before A's terminal result reaches Events()
// (the Claude adapter woke its cancel waiters before the channel send). The
// user sends turn B at once. A's late EventResult must not end turn B: B stays
// active until B's own result.
func TestStopInteractiveSession_LateCancelResultDoesNotFinishNextTurn(t *testing.T) {
	inner := newControllableSession("cancel-handoff-1")
	bPrompt := make(chan struct{})
	var bOnce sync.Once
	sess := &promptHookSession{controllableAgentSession: inner, onSend: func(prompt string) {
		if strings.Contains(prompt, "turn-B") {
			bOnce.Do(func() { close(bPrompt) })
		}
	}}
	agent := &controllableAgent{nextSession: sess}
	p := &gatedFirstReplyPlatform{
		stubPlatformEngine: stubPlatformEngine{n: "test"},
		gateText:           "starting",
		entered:            make(chan struct{}),
		release:            make(chan struct{}),
	}
	e := NewEngine("test", agent, []Platform{p}, "", LangEnglish)
	e.SetInstantReply(InstantReplyCfg{Enabled: true, Content: "starting"})

	key := "test:user1"
	session := e.sessions.GetOrCreateActive(key)
	session.SetAgentSessionID(inner.sessionID, agent.Name())

	// A's terminal result arrives late: after B's prompt, or after 300ms when
	// no B prompt comes. A's consumer leaves its platform call only after that
	// result was read by someone, or after 300ms.
	lateSent := make(chan struct{})
	inner.cancelTurnFn = func() error {
		go func() {
			select {
			case <-bPrompt:
			case <-time.After(300 * time.Millisecond):
			}
			inner.events <- Event{Type: EventResult, Done: true}
			close(lateSent)
			deadline := time.Now().Add(300 * time.Millisecond)
			for len(inner.events) > 0 && time.Now().Before(deadline) {
				time.Sleep(5 * time.Millisecond)
			}
			close(p.release)
		}()
		return nil
	}

	e.handleMessage(p, &Message{SessionKey: key, ReplyCtx: "ctx", Content: "turn-A", Platform: "test", UserID: "user1", MessageID: "m-a"})
	select {
	case <-p.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("turn A did not reach its platform call")
	}

	stopDone := make(chan bool, 1)
	go func() { stopDone <- e.stopInteractiveSession(key, p, "ctx") }()
	select {
	case stopped := <-stopDone:
		if !stopped {
			t.Fatal("stopInteractiveSession returned false; want the graceful cancel path")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("/stop did not return")
	}
	if got := inner.cancelCalls.Load(); got != 1 {
		t.Fatalf("CancelTurn called %d times, want 1", got)
	}

	e.handleMessage(p, &Message{SessionKey: key, ReplyCtx: "ctx", Content: "turn-B", Platform: "test", UserID: "user1", MessageID: "m-b"})
	select {
	case <-bPrompt:
	case <-time.After(3 * time.Second):
		t.Fatal("turn B's prompt was not sent")
	}
	select {
	case <-lateSent:
	case <-time.After(3 * time.Second):
		t.Fatal("A's late result was not delivered")
	}

	// B has no result yet, so B must still hold the busy lock.
	deadline := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(deadline) {
		if !session.Busy() {
			t.Fatal("turn B ended before its own result: A's late cancel result finished it")
		}
		time.Sleep(10 * time.Millisecond)
	}

	inner.events <- Event{Type: EventResult, Content: "answer-B", Done: true}
	deadline = time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		delivered := false
		for _, s := range p.getSent() {
			if strings.Contains(s, "answer-B") {
				delivered = true
			}
		}
		if delivered && !session.Busy() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("turn B did not deliver its answer and release the lock; sent = %q, busy = %v", p.getSent(), session.Busy())
}
