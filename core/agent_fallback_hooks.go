package core

import (
	"errors"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"
)

// Agent fallback chain: the engine side (agent_fallback.go has the chain).
//
// Fork-only (linkendr/cc-connect). engine.go is upstream's file and changes
// often in the same functions the chain hooks into (processInteractiveMessageWith,
// processInteractiveEvents, drainPendingMessages, /stop, cleanup). To keep the
// daily upstream merge clean, engine.go holds only short calls into this
// file; every fallback decision and every fork-only field is here. Without a
// chain (agentFallback == nil) each hook does what upstream does.
//
// Not covered: relay turns (HandleRelay) do not use the chain; a primary
// usage-limit error is returned to the caller.

// interactiveFallback holds the fork's fields of interactiveState (embedded
// there). Guarded by interactiveState.mu unless noted.
type interactiveFallback struct {
	// fallbackIntercept selects which failures processInteractiveEvents
	// hands back to its caller.
	fallbackIntercept fallbackIntercept
	// turnInput is the message of the in-flight turn, so the caller can
	// answer it with a fallback agent.
	turnInput queuedMessage
	// turnCCKey is the CC_SESSION_KEY of the in-flight turn's caller ("" for
	// a queued turn the drain started). The event loop's hand-back builds
	// the turn's fallbackGuard from its own arguments and this key.
	turnCCKey string
	// fallbackRun is where the event loop of one fallback turn
	// (runFallbackTurn) leaves its outcome.
	fallbackRun fallbackRunOutcome
	// isFallbackState marks the state of a fallback agent (no
	// auto-compress, no primary write-back mark).
	isFallbackState bool
	// dispatchCh serializes guarded fallback-chain sends against /stop
	// (guardedDispatch); stopRequested (atomic) is raised by /stop before it
	// takes that gate. Both unused without a fallback chain.
	dispatchCh    chan struct{}
	stopRequested atomic.Bool
	// unsolicitedLingering is the done channel of an unsolicited reader that
	// did not exit within unsolicitedReaderStopTimeout (nil = none known).
	unsolicitedLingering chan struct{}
}

// permissionClaim is embedded in pendingPermission. claimed is taken (CAS)
// by the one path that answers the request when an agent fallback chain is
// configured: the user's reply, engine Stop, or the shutdown branch of the
// wait. The loser does not answer.
type permissionClaim struct {
	claimed atomic.Bool
}

// claim reports whether the caller won the right to answer the request.
func (pc *permissionClaim) claim() bool {
	return pc.claimed.CompareAndSwap(false, true)
}

// dispatchGate returns the state's one-slot dispatch gate, creating it on
// first use.
func (s *interactiveState) dispatchGate() chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dispatchCh == nil {
		s.dispatchCh = make(chan struct{}, 1)
	}
	return s.dispatchCh
}

// sessionCloseWaitBudget is awaitSessionClose's budget (upstream: a literal
// closeTimeout + 30s). A var only so tests can shorten it.
var sessionCloseWaitBudget = closeTimeout + 30*time.Second

// installFallbackHookMapper runs in SetHooks.
func (e *Engine) installFallbackHookMapper() {
	if e.agentFallback != nil {
		e.hooks.setEventMapper(e.agentFallback.hookKeyMapper)
	}
}

// denyPendingPermissionsOnStop runs in engine Stop on the removed states: a
// turn (primary or fallback) waiting for a permission answer must not
// outlive shutdown; deny it so the wait returns, and drop hand-off markers.
func (e *Engine) denyPendingPermissionsOnStop(states map[string]*interactiveState) {
	if e.agentFallback == nil {
		return
	}
	for key, state := range states {
		state.mu.Lock()
		pending := state.pending
		state.pending = nil
		as := state.agentSession
		state.mu.Unlock()
		if pending == nil {
			continue
		}
		if pending.claim() && as != nil {
			if err := as.RespondPermission(pending.RequestID, PermissionResult{Behavior: "deny", Message: "cc-connect is shutting down"}); err != nil {
				slog.Debug("engine.Stop: deny pending permission", "session", key, "error", err)
			}
		}
		pending.resolve()
	}
	e.agentFallback.clearHandoffs()
}

// claimPermissionAnswer reports whether the user's AskUserQuestion answer
// may answer pending (false: engine shutdown already answered it).
func (e *Engine) claimPermissionAnswer(pending *pendingPermission) bool {
	return e.agentFallback == nil || pending.claim()
}

// claimPermissionReply is claimPermissionAnswer for an allow/deny/approve-all
// reply (lower is the lower-cased reply). Other text claims nothing.
func (e *Engine) claimPermissionReply(pending *pendingPermission, lower string) bool {
	if e.agentFallback == nil {
		return true
	}
	if !isApproveAllResponse(lower) && !isAllowResponse(lower) && !isDenyResponse(lower) {
		return true
	}
	return pending.claim()
}

// mustStartFresh is the unsafe-resume decision of
// getOrCreateInteractiveStateWith. Without a chain it is upstream's
// condition. With a chain the unsafe-resume marker is settled on every
// start, including a fresh start and an expired teardown wait, so a later
// clean restart does not discard a good conversation.
func (e *Engine) mustStartFresh(sessionKey, startSessionID string, closeSettled bool) bool {
	if e.agentFallback == nil {
		return startSessionID != "" && (!closeSettled || e.consumeUnsafeResume(sessionKey))
	}
	marked := e.consumeUnsafeResume(sessionKey)
	return startSessionID != "" && (!closeSettled || marked)
}

// fallbackEarlyClose is a teardown registered by cleanupInteractiveState
// before the session pointer is detached, under the same interactiveMu hold,
// so a /stop or a respawn that sees the detached state also sees the
// teardown (awaitSessionClose) and never resumes into the still-running
// process. Zero value: none (no chain, or no session).
type fallbackEarlyClose struct {
	finish func()
}

// beginFallbackEarlyClose runs in cleanupInteractiveState with interactiveMu
// held.
func (e *Engine) beginFallbackEarlyClose(sessionKey string, state *interactiveState) fallbackEarlyClose {
	if e.agentFallback == nil || state == nil {
		return fallbackEarlyClose{}
	}
	state.mu.Lock()
	hasSession := state.agentSession != nil
	state.mu.Unlock()
	if !hasSession {
		return fallbackEarlyClose{}
	}
	return fallbackEarlyClose{finish: e.beginSessionClose(sessionKey)}
}

// close closes agentSession and ends the registered teardown. It reports
// false when no teardown was registered (the caller closes upstream's way).
func (c fallbackEarlyClose) close(e *Engine, sessionKey string, agentSession AgentSession, platform Platform, replyCtx any) bool {
	if c.finish == nil {
		return false
	}
	e.closeAgentSession(sessionKey, agentSession, platform, replyCtx)
	c.finish()
	return true
}

// fallbackCleanupDetached runs in cleanupInteractiveState right after the
// session was detached (test hook only).
func (e *Engine) fallbackCleanupDetached(sessionKey string) {
	if e.agentFallback != nil && testHookCleanupDetached != nil {
		testHookCleanupDetached(sessionKey)
	}
}

// cleanupFallbackStatesAfterRemoval runs at the end of
// cleanupInteractiveState: it also closes the conversation's fallback states.
func (e *Engine) cleanupFallbackStatesAfterRemoval(sessionKey string) {
	if e.agentFallback != nil && testHookAfterStateRemoved != nil {
		testHookAfterStateRemoved(sessionKey)
	}
	e.cleanupFallbackStates(sessionKey)
}

// stopUnsolicitedReaderConfirmed stops the unsolicited reader like
// stopUnsolicitedReader and reports whether no reader goroutine is left
// running — including one that outlived an earlier stop timeout. Used by
// the agent fallback chain before it hands Events() to a queued turn.
func (e *Engine) stopUnsolicitedReaderConfirmed(state *interactiveState) bool {
	e.stopUnsolicitedReader(state)
	state.mu.Lock()
	lingering := state.unsolicitedLingering
	state.mu.Unlock()
	if lingering == nil {
		return true
	}
	select {
	case <-lingering:
		state.mu.Lock()
		if state.unsolicitedLingering == lingering {
			state.unsolicitedLingering = nil
		}
		state.mu.Unlock()
		return true
	default:
		return false
	}
}

// primaryFallbackTurn is the chain's view of one foreground turn of
// processInteractiveMessageWith. chain is nil when the turn has no fallback
// (no chain configured, or a multi-workspace agent).
type primaryFallbackTurn struct {
	e            *Engine
	chain        *agentFallbackChain
	guard        fallbackGuard
	agent        Agent
	workspaceDir string
	// skipPrimary: the primary reported a limit reset time that has not
	// passed; answer with the chain without sending to the primary.
	skipPrimary  bool
	blockedCause turnOutcome
}

func (e *Engine) beginPrimaryFallbackTurn(p Platform, msg *Message, agent Agent, state *interactiveState, session *Session, sessions *SessionManager, interactiveKey, workspaceDir, ccSessionKey string, lockGen uint64) *primaryFallbackTurn {
	t := &primaryFallbackTurn{
		e:            e,
		guard:        fallbackGuard{primary: state, session: session, sessions: sessions, lockGen: lockGen, interactiveKey: interactiveKey, ccSessionKey: ccSessionKey},
		agent:        agent,
		workspaceDir: workspaceDir,
	}
	if e.agentFallback == nil {
		return t
	}
	chained := agent == e.agent
	state.mu.Lock()
	state.turnCCKey = ccSessionKey
	if chained {
		// The primary's fallback policy belongs to the state, not to this
		// turn: messages queued behind a fallback-only turn must still be
		// eligible (blocked check, interception) when they are drained.
		state.fallbackIntercept = interceptRetriable
		state.turnInput = queuedMessageFromMessage(p, msg)
	}
	state.mu.Unlock()
	if !chained {
		return t
	}
	t.chain = e.agentFallback
	if until, kind, blocked := t.chain.primaryBlocked(); blocked {
		t.skipPrimary = true
		t.blockedCause = turnOutcome{Intercepted: true, Kind: kind, ResetAt: until, blocked: true, Input: queuedMessageFromMessage(p, msg)}
	}
	return t
}

// answerWithoutPrimary runs a skipPrimary turn to its end, in place of the
// rest of processInteractiveMessageWith after the unsolicited reader was
// stopped. It mirrors upstream's tail of that function (restart a queue
// left by a dead agent, arm the idle close or schedule reader recovery,
// drain queued messages and start the unsolicited reader once the queue is
// empty) and returns true when the session lock was released.
func (t *primaryFallbackTurn) answerWithoutPrimary(needResync bool, stopTyping *func()) bool {
	e, state, g := t.e, t.guard.primary, t.guard
	if needResync && state.agentSession != nil {
		drainEvents(state.agentSession.Events())
	}
	// The fallback turn runs its own typing indicator.
	if *stopTyping != nil {
		(*stopTyping)()
		*stopTyping = nil
	}
	e.answerWithFallbacks(g, t.blockedCause)

	e.interactiveMu.Lock()
	replacement := e.interactiveStates[g.interactiveKey]
	e.interactiveMu.Unlock()
	if replacement != nil && replacement != state {
		replacement.mu.Lock()
		queued := len(replacement.pendingMessages) > 0
		replacement.mu.Unlock()
		if queued {
			e.drainOrphanedQueue(g.session, g.sessions, g.interactiveKey, t.agent, t.workspaceDir, g.lockGen)
			return true
		}
	}

	state.mu.Lock()
	alive := state.agentSession != nil && state.agentSession.Alive() && !state.stopped && !state.eventsNeedResync
	state.mu.Unlock()
	if alive {
		e.scheduleAgentSessionIdleClose(g.interactiveKey, state)
	} else {
		e.scheduleUnsolicitedReaderRecovery(state, g.session, g.sessions, g.interactiveKey, t.workspaceDir)
	}
	return e.drainPendingMessages(state, g.session, g.sessions, g.interactiveKey, g.lockGen, func() {
		e.startIdleUnsolicitedReader(state, g.session, g.sessions, g.interactiveKey, t.workspaceDir)
	})
}

// withWriteback prepends the fallback turns the primary has not seen.
func (t *primaryFallbackTurn) withWriteback(prompt string) string {
	if t.chain == nil {
		return prompt
	}
	return primaryWritebackPrefix(t.guard.session) + prompt
}

// fallbackEvents is the chain's view of one processInteractiveEvents run.
// A hook that returns true has ended the turn; the caller returns at once.
// cp, sp and pendingSend point at the event loop's own variables, which it
// replaces when it picks up a queued turn. outcome belongs to this run only:
// handBack, the run's first deferred call, answers it when the run returns.
type fallbackEvents struct {
	e               *Engine
	state           *interactiveState
	session         *Session
	sessions        *SessionManager
	sessionKey      string
	lockGen         uint64
	ccSessionKey    string
	cp              **compactProgressWriter
	sp              **streamPreview
	pendingSend     *<-chan error
	outcome         turnOutcome
	runOut          *turnOutcome
	intercept       fallbackIntercept
	isFallbackState bool
}

// fallbackRunOutcome is set by runFallbackTurn before it starts the event
// loop: the run whose send channel is sendDone leaves its outcome in *out.
type fallbackRunOutcome struct {
	sendDone <-chan error
	out      *turnOutcome
}

// beginFallbackEvents runs at the start of processInteractiveEvents, before
// its other deferred calls, and takes what the caller set on the state for
// this run.
func (e *Engine) beginFallbackEvents(state *interactiveState, session *Session, sessions *SessionManager, sessionKey string, lockGen uint64, sendDone <-chan error) *fallbackEvents {
	f := &fallbackEvents{e: e, state: state, session: session, sessions: sessions, sessionKey: sessionKey, lockGen: lockGen}
	state.mu.Lock()
	f.intercept = state.fallbackIntercept
	f.isFallbackState = state.isFallbackState
	f.ccSessionKey = state.turnCCKey
	if sendDone != nil && state.fallbackRun.sendDone == sendDone {
		f.runOut = state.fallbackRun.out
		state.fallbackRun = fallbackRunOutcome{}
	}
	state.mu.Unlock()
	return f
}

// bindTurn gives the hooks the event loop's writer and send variables.
func (f *fallbackEvents) bindTurn(cp **compactProgressWriter, sp **streamPreview, pendingSend *<-chan error) {
	f.cp, f.sp, f.pendingSend = cp, sp, pendingSend
}

// handBack runs when the event loop returns, after the loop's own deferred
// calls (typing stop, done reaction). A fallback agent's run leaves its
// outcome to runFallbackTurn. A primary turn, or a queued turn of the drain,
// that the loop handed back is answered by the fallback chain, with the
// guard of this run: its own state, session and lock generation.
func (f *fallbackEvents) handBack() {
	if f.isFallbackState {
		if f.runOut == nil {
			slog.Error("agent fallback: fallback run has no outcome slot; its outcome is lost",
				"session_key", f.sessionKey, "intercepted", f.outcome.Intercepted)
			return
		}
		*f.runOut = f.outcome
		return
	}
	if f.outcome.Intercepted {
		f.e.answerWithFallbacks(fallbackGuard{
			primary: f.state, session: f.session, sessions: f.sessions, lockGen: f.lockGen,
			interactiveKey: f.sessionKey, ccSessionKey: f.ccSessionKey,
		}, f.outcome)
	}
}

// interceptTurn ends the turn without showing the failure and hands it to
// handBack.
func (f *fallbackEvents) interceptTurn(kind ErrorKind, resetAt time.Time, errText string) {
	e := f.e
	// First, before any platform call or wait below can block: from here
	// until the chain is done the conversation counts as live for the
	// stale-lock self-heal (answerWithFallbacks ends the hand-off).
	handoff := false
	if f.intercept == interceptRetriable && e.agentFallback != nil {
		e.agentFallback.beginHandoff(f.sessionKey)
		handoff = true
	}
	(*f.cp).Finalize(ProgressCardStateFailed)
	(*f.sp).discard()
	if *f.pendingSend != nil {
		if err := <-*f.pendingSend; err != nil {
			slog.Debug("async send error during intercepted failure", "error", err)
		}
		*f.pendingSend = nil
	}
	f.state.mu.Lock()
	in := f.state.turnInput
	f.state.mu.Unlock()
	slog.Warn("agent turn failed; handing it to the fallback chain",
		"session_key", f.sessionKey, "kind", kind, "reset_at", resetAt, "error", errText)
	f.outcome = turnOutcome{Intercepted: true, Kind: kind, ResetAt: resetAt, ErrText: errText, Input: in, handoffBegun: handoff}
}

// sendFailed handles the result of the turn's Send: a guarded send that found
// the turn stopped ends it silently; a non-last fallback agent hands any
// send error to the next one.
func (f *fallbackEvents) sendFailed(err error) bool {
	if errors.Is(err, errTurnCancelled) {
		(*f.sp).discard()
		return true
	}
	if err != nil && f.intercept == interceptAnyError {
		f.interceptTurn(ErrorKindUnknown, time.Time{}, f.e.agentErrorText(err))
		return true
	}
	return false
}

func (f *fallbackEvents) beforePermissionPublish() {
	if f.e.agentFallback != nil && testHookBeforePermissionPublish != nil {
		testHookBeforePermissionPublish()
	}
}

// awaitPermission waits for the permission answer. With a chain, engine
// shutdown also ends the wait, including a request published after Stop
// took its snapshot of pending permissions (deny it).
func (f *fallbackEvents) awaitPermission(pending *pendingPermission) {
	e, state := f.e, f.state
	if e.agentFallback == nil {
		<-pending.Resolved
		return
	}
	select {
	case <-pending.Resolved:
	case <-e.ctx.Done():
		// Answer only if nobody else took the request (Stop's snapshot or a
		// user reply clear state.pending first).
		state.mu.Lock()
		owned := state.pending == pending
		if owned {
			state.pending = nil
		}
		as := state.agentSession
		state.mu.Unlock()
		if owned && pending.claim() && as != nil {
			_ = as.RespondPermission(pending.RequestID, PermissionResult{Behavior: "deny", Message: "cc-connect is shutting down"})
		}
		pending.resolve()
	}
}

// keepAgentSessionID keeps tracking the agent session ID of a turn that is
// handed back, so the primary's --resume stays correct.
func (f *fallbackEvents) keepAgentSessionID() {
	state, session := f.state, f.session
	if state.agentSession == nil {
		return
	}
	if currentID := state.agentSession.CurrentSessionID(); currentID != "" && session.GetAgentSessionID() != currentID {
		session.SetAgentSessionID(currentID, f.e.agent.Name())
		f.sessions.Save()
	}
}

// resultIntercepted hands back a turn whose EventResult reports a failure at
// the API layer (e.g. Claude's usage cap); the process is alive and the turn
// is over.
func (f *fallbackEvents) resultIntercepted(event Event) bool {
	if !f.intercept.catchesResult(event) {
		return false
	}
	f.keepAgentSessionID()
	f.state.mu.Lock()
	f.state.eventsNeedResync = false
	f.state.mu.Unlock()
	f.interceptTurn(event.ErrorKind, event.ErrorResetAt, event.Content)
	return true
}

// errorIntercepted hands back a turn that ended with a matching EventError.
func (f *fallbackEvents) errorIntercepted(event Event) bool {
	if !f.intercept.catchesError(event) {
		return false
	}
	f.keepAgentSessionID()
	f.state.mu.Lock()
	f.state.eventsNeedResync = true
	f.state.mu.Unlock()
	errText := ""
	if event.Error != nil {
		errText = f.e.agentErrorText(event.Error)
	}
	f.interceptTurn(event.ErrorKind, event.ErrorResetAt, errText)
	return true
}

// answered runs when the agent answered a turn. When the primary answers,
// it has now seen every history entry, including any fallback turns written
// back into this prompt.
func (f *fallbackEvents) answered() {
	if f.e.agentFallback != nil && !f.isFallbackState {
		f.session.MarkPrimarySeen()
	}
}

// processExited handles the agent's event channel closing (process exit)
// for a fallback agent that gave no answer.
func (f *fallbackEvents) processExited(textParts int, replyCtx any) bool {
	e, state := f.e, f.state
	if f.intercept == interceptAnyError && textParts == 0 {
		// A fallback agent died without answering: let the next one try.
		state.mu.Lock()
		state.eventsNeedResync = true
		state.mu.Unlock()
		e.cleanupInteractiveState(f.sessionKey, state)
		f.interceptTurn(ErrorKindUnknown, time.Time{}, e.i18n.T(MsgAgentProcessExited))
		return true
	}
	if f.isFallbackState && textParts == 0 && !state.isStopped() {
		// The last fallback agent exited without an answer or an error
		// event: say so, or the user sees only the fallback notice.
		(*f.cp).Finalize(ProgressCardStateFailed)
		(*f.sp).discard()
		state.mu.Lock()
		state.eventsNeedResync = true
		p := state.platform
		state.mu.Unlock()
		e.cleanupInteractiveState(f.sessionKey, state)
		e.send(p, replyCtx, fmt.Sprintf(e.i18n.T(MsgError), e.i18n.T(MsgAgentFallbackNoAnswer)))
		return true
	}
	return false
}

// fallbackDrainLostTurnLocked reports (state.mu held, in the same hold as
// the queue pop) that this drain no longer owns the turn: a nested drain
// released generation lockGen and a newer turn took the lock, or /stop
// released it. The drain then leaves the queue to the current owner; it
// never pops work of another generation.
func (e *Engine) fallbackDrainLostTurnLocked(state *interactiveState, session *Session, lockGen uint64) bool {
	return e.agentFallback != nil && state.fallbackIntercept == interceptRetriable && !session.HoldsLock(lockGen)
}

// fallbackDrainTurn is the chain's view of one queued turn of
// drainPendingMessages. When active is false every method does nothing and
// the loop is upstream's code.
type fallbackDrainTurn struct {
	e          *Engine
	state      *interactiveState
	session    *Session
	sessions   *SessionManager
	sessionKey string
	lockGen    uint64
	queued     queuedMessage
	active     bool
}

// beginFallbackDrainTurnLocked runs with state.mu held, after queued was
// popped.
func (e *Engine) beginFallbackDrainTurnLocked(state *interactiveState, session *Session, sessions *SessionManager, sessionKey string, lockGen uint64, queued queuedMessage) *fallbackDrainTurn {
	state.turnInput = queued
	state.turnCCKey = ""
	return &fallbackDrainTurn{
		e: e, state: state, session: session, sessions: sessions, sessionKey: sessionKey, lockGen: lockGen, queued: queued,
		active: state.fallbackIntercept == interceptRetriable && e.agentFallback != nil,
	}
}

func (d *fallbackDrainTurn) guard() fallbackGuard {
	return fallbackGuard{primary: d.state, session: d.session, sessions: d.sessions, lockGen: d.lockGen, interactiveKey: d.sessionKey}
}

// turnLive reports whether this drain still owns the user's turn: /stop
// marks the state stopped and force-unlocks the session.
func (d *fallbackDrainTurn) turnLive(st *interactiveState) bool {
	return !st.isStopped() && d.session.HoldsLock(d.lockGen)
}

// answeredWithoutPrimary: while the primary is inside a known limit window,
// queued messages go straight to the fallback agents.
func (d *fallbackDrainTurn) answeredWithoutPrimary() bool {
	if !d.active {
		return false
	}
	e := d.e
	until, kind, blocked := e.agentFallback.primaryBlocked()
	if !blocked {
		return false
	}
	d.session.AddHistory("user", d.queued.content)
	d.sessions.Save()
	// Like a normal turn: no idle close of the primary while this turn runs
	// (its close would stop the fallback too); re-armed when the turn is
	// done.
	e.cancelAgentSessionIdleClose(d.state)
	e.answerWithFallbacks(d.guard(), turnOutcome{Intercepted: true, Kind: kind, ResetAt: until, blocked: true, Input: d.queued})
	e.rearmPrimaryIdleClose(d.state, d.sessionKey)
	return true
}

// ensurePrimary checks that the drain still owns the turn and, when the
// primary died on a retriable failure (e.g. claude exited on a 429), starts
// it again for this queued message instead of dropping the queue; the new
// state adopts the remaining queue and keeps fallback eligibility. It
// replaces *state and *as with the restarted ones. false: the turn is gone
// (the user was told); the drain returns.
func (d *fallbackDrainTurn) ensurePrimary(state **interactiveState, as *AgentSession) bool {
	if !d.active {
		return true
	}
	e, queued, st := d.e, d.queued, *state
	if !d.turnLive(st) {
		e.send(queued.platform, queued.replyCtx, fmt.Sprintf(e.i18n.T(MsgError), e.i18n.T(MsgSessionReset)))
		return false
	}
	if *as != nil && (*as).Alive() {
		return true
	}
	restarted := e.getOrCreateInteractiveStateWith(d.sessionKey, queued.platform, queued.replyCtx, d.session, d.sessions, nil, "")
	if !d.turnLive(st) {
		// /stop ran while the primary was starting: this turn is gone, and
		// the state may belong to a newer turn.
		e.send(queued.platform, queued.replyCtx, fmt.Sprintf(e.i18n.T(MsgError), e.i18n.T(MsgSessionReset)))
		return false
	}
	restarted.mu.Lock()
	restarted.platform = queued.platform
	restarted.replyCtx = queued.replyCtx
	restarted.currentMessageID = queued.messageID
	restarted.fromVoice = queued.fromVoice
	restarted.currentTurnUserMessageTimeMs = queued.userMessageTimeMs
	restarted.fallbackIntercept = interceptRetriable
	restarted.turnInput = queued
	restarted.turnCCKey = ""
	*as = restarted.agentSession
	restarted.mu.Unlock()
	*state = restarted
	d.state = restarted
	return true
}

// ownEvents makes this goroutine the owner of the event channel for the
// queued turn: it stops the unsolicited reader first, or the reader can
// steal the turn's result or auto-deny its permission requests. If the
// reader does not confirm its exit, it fails closed for this message
// (false: the drain skips it).
func (d *fallbackDrainTurn) ownEvents() bool {
	if !d.active {
		return true
	}
	e, queued := d.e, d.queued
	if !e.stopUnsolicitedReaderConfirmed(d.state) {
		slog.Warn("agent fallback: unsolicited reader still running; not sending queued message",
			"session", d.sessionKey)
		e.send(queued.platform, queued.replyCtx, e.i18n.T(MsgPreviousProcessing))
		return false
	}
	e.cancelAgentSessionIdleClose(d.state)
	return true
}

// beforeSend prepends the fallback turns the primary has not seen and wraps
// as so its Send is a guarded dispatch (mutually exclusive with /stop).
func (d *fallbackDrainTurn) beforeSend(prompt string, as AgentSession) (string, AgentSession) {
	if !d.active {
		return prompt, as
	}
	sendState := d.state
	guarded := &guardedSendSession{
		AgentSession: as,
		owner:        sendState,
		live:         func() bool { return d.turnLive(sendState) },
	}
	return primaryWritebackPrefix(d.session) + prompt, guarded
}

// afterTurn runs after a queued turn (the event loop has already answered a
// turn it handed back): for a fallback-eligible drain, it hands the event
// channel back to the unsolicited reader.
func (d *fallbackDrainTurn) afterTurn() {
	if d.active {
		d.e.restartUnsolicitedReaderAfterQueuedTurn(d.state, d.session, d.sessions, d.sessionKey)
	}
}

// guardedSendSession is the primary's session for one queued turn: Send
// runs through guardedDispatch.
type guardedSendSession struct {
	AgentSession
	owner *interactiveState
	live  func() bool
}

func (g *guardedSendSession) Send(prompt string, messageID string, images []ImageAttachment, files []FileAttachment) error {
	return guardedDispatch(g.owner, g.live, func() error {
		return g.AgentSession.Send(prompt, messageID, images, files)
	})
}

// rearmPrimaryIdleClose re-arms the primary's idle close after a queued
// fallback-only turn, under the same conditions as the end of a turn.
func (e *Engine) rearmPrimaryIdleClose(state *interactiveState, sessionKey string) {
	state.mu.Lock()
	alive := state.agentSession != nil && state.agentSession.Alive() && !state.stopped && !state.eventsNeedResync
	state.mu.Unlock()
	if alive {
		e.scheduleAgentSessionIdleClose(sessionKey, state)
	}
}

// restartUnsolicitedReaderAfterQueuedTurn hands the event channel back to
// the unsolicited reader and re-arms the idle close after a queued turn,
// mirroring the end of processInteractiveMessageWith. The next queued turn
// stops the reader again before it sends.
func (e *Engine) restartUnsolicitedReaderAfterQueuedTurn(state *interactiveState, session *Session, sessions *SessionManager, sessionKey string) {
	state.mu.Lock()
	alive := state.agentSession != nil && state.agentSession.Alive() && !state.stopped && !state.eventsNeedResync
	workspaceDir := state.workspaceDir
	state.mu.Unlock()
	if alive {
		e.startUnsolicitedReader(state, session, sessions, sessionKey, workspaceDir)
		e.scheduleAgentSessionIdleClose(sessionKey, state)
	}
}

// stopTarget is the state one stopInteractiveSessionOnce call may stop.
// nil (or a nil expect) stops whatever state the key maps to.
type stopTarget struct {
	expect *interactiveState
	retry  bool // set when the key maps to another state
}

// retarget reports (and records in retry) that the key now maps to a state
// other than the expected one.
func (t *stopTarget) retarget(state *interactiveState) bool {
	if t == nil || t.expect == nil || state == t.expect {
		return false
	}
	t.retry = true
	return true
}

// maxStopRetargets bounds how often /stop re-resolves a replaced state.
const maxStopRetargets = 8

// stopInteractiveSessionWithOptions stops the session's turn. Without a
// chain it is upstream's function (stopInteractiveSessionOnce in engine.go).
// With a chain it makes the stop mutually exclusive with a guarded send for
// the turn (see guardedDispatch): take the gate of one resolved state, then
// stop that same state; if the key now maps to a different state (e.g. a
// dead primary was restarted meanwhile), retry on it.
func (e *Engine) stopInteractiveSessionWithOptions(sessionKey string, notifyQueued bool) bool {
	if e.agentFallback == nil {
		return e.stopInteractiveSessionOnce(sessionKey, notifyQueued, nil)
	}
	for attempt := 0; ; attempt++ {
		target, release := e.acquireDispatchForStop(sessionKey)
		if target == nil {
			// No primary state (e.g. its idle close removed it): still stop
			// fallback agents answering for this conversation.
			stopped := false
			for _, key := range e.agentFallback.ownedStatesOf(sessionKey) {
				if e.stopInteractiveSessionWithOptions(key, notifyQueued) {
					stopped = true
				}
			}
			return stopped
		}
		t := &stopTarget{expect: target}
		if attempt >= maxStopRetargets {
			t.expect = nil // keep stopping whatever is there; do not spin
		}
		stopped := e.stopInteractiveSessionOnce(sessionKey, notifyQueued, t)
		if release != nil {
			release()
		}
		if !t.retry {
			return stopped
		}
	}
}

// acquireDispatchForStop resolves the session's state once, raises its
// stopRequested and takes its dispatch gate (bounded by stopDispatchWait).
// Returns that state (nil = none) and the release func (nil when the wait
// timed out).
func (e *Engine) acquireDispatchForStop(sessionKey string) (*interactiveState, func()) {
	e.interactiveMu.Lock()
	st := e.interactiveStates[sessionKey]
	e.interactiveMu.Unlock()
	if st == nil {
		return nil, nil
	}
	if hook := testHookStopAfterCapture; hook != nil {
		hook()
	}
	st.stopRequested.Store(true)
	gate := st.dispatchGate()
	select {
	case gate <- struct{}{}:
		return st, func() { <-gate }
	case <-time.After(stopDispatchWait):
		slog.Warn("stop: dispatch still inside Send; stopping the running turn", "session_key", sessionKey)
		return st, nil
	}
}

// awaitDetachedFallbackClose runs in /stop's normal cleanup. When the
// session was already detached by a teardown in flight (e.g. a fallback
// retirement), it waits for that teardown (bounded by awaitSessionClose)
// before the busy lock is released, so the next turn cannot start while
// that process may still be alive.
func (e *Engine) awaitDetachedFallbackClose(sessionKey string, agentSession AgentSession) {
	if e.agentFallback != nil && agentSession == nil {
		e.awaitSessionClose(sessionKey)
	}
}
