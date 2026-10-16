package core

import (
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// Agent fallback chain.
//
// When the project's primary agent ends a turn with a retriable backend error
// (ErrorKind.IsRetriable: rate limit, overloaded, usage limit), the engine
// answers the same message with the next agent of an ordered chain, in the
// same chat/thread, after a one-line notice. The chain is turn-scoped: each
// new message starts on the primary again, unless the primary reported when
// its limit resets — then fallback[0] answers directly until that time.
//
// The fallback only decides which agent CLI handles a turn. It never changes
// how the primary agent is started or how it reaches its backend.
//
// Each fallback agent keeps its own interactive state (key
// "<interactiveKey>#fallback:<i>") and its own in-memory Session, keyed by the
// primary cc-connect Session ID, so the primary's stored agent session ID
// (Claude's --resume target) is never touched and /new gives the fallback a
// fresh conversation too. Context comes from cc-connect's own session history
// (Session.History): the last fallbackHistoryEntries user/assistant messages
// the fallback agent has not seen yet are prepended to the prompt as plain
// text. A successful fallback answer is appended to the primary session's
// history, so /history and the next hand-off include it.
//
// Not covered (TODO): multi-workspace projects (rejected at config load) and
// the relay/bot-to-bot path (relay.go), which sends to the agent outside
// processInteractiveMessageWith. Every path that goes through
// processInteractiveMessageWith — platform messages, queued messages, cron,
// timers, heartbeat, webhooks — uses the chain.

// AgentFallbackSpec describes one fallback agent. cmd/cc-connect builds it
// from [projects.agent].fallback.
type AgentFallbackSpec struct {
	Type    string         // registered agent type, passed to CreateAgent
	Options map[string]any // factory options (model, work_dir, mode, …)
	Label   string         // name shown in the notice; "" = display name + model
}

// AgentDisplayNamer is an optional Agent capability: the product name shown to
// users in fallback notices (e.g. "Claude Code", "OpenCode").
type AgentDisplayNamer interface {
	DisplayName() string
}

const (
	// fallbackHistoryEntries is how many cc-connect history messages (user or
	// assistant) are handed to a fallback agent as context.
	fallbackHistoryEntries = 6
	// fallbackHistoryEntryMaxRunes truncates each handed-off message.
	fallbackHistoryEntryMaxRunes = 2000
	// maxPrimaryBlock caps a parsed reset time, so a misparsed date cannot
	// park the primary agent for longer than a weekly limit window.
	maxPrimaryBlock = 8 * 24 * time.Hour
)

// fallbackIntercept tells processInteractiveEvents which failures to hand
// back to the caller (without showing them) instead of surfacing them.
type fallbackIntercept uint8

const (
	interceptNone      fallbackIntercept = iota // default: surface everything (no fallback, or last fallback)
	interceptRetriable                          // primary agent: hand back retriable backend errors
	interceptAnyError                           // non-last fallback agent: hand back any failure
)

func (m fallbackIntercept) catchesResult(ev Event) bool {
	return m != interceptNone && ev.ErrorKind.IsRetriable()
}

func (m fallbackIntercept) catchesError(ev Event) bool {
	switch m {
	case interceptRetriable:
		return ev.ErrorKind.IsRetriable()
	case interceptAnyError:
		return true
	}
	return false
}

// turnOutcome is what processInteractiveEvents reports to its caller. The
// zero value means the turn was handled (answered, or its error shown).
type turnOutcome struct {
	// Intercepted is true when the turn failed and nothing was shown to the
	// user; the caller must answer Input some other way.
	Intercepted bool
	Kind        ErrorKind
	ResetAt     time.Time
	ErrText     string
	// Input is the message whose turn failed. It may be a queued message
	// that the event loop picked up after the original one.
	Input queuedMessage
	// Cancelled is true when the turn was stopped (/stop, session reset,
	// shutdown); nothing more must run for it.
	Cancelled bool
	// Failed is true when the turn ended with an error already shown.
	Failed bool
	// blocked marks a turn that skipped the primary because of an earlier
	// reset time (ResetAt).
	blocked bool
}

// fallbackGuard is the turn-scoped context of a fallback run: the primary
// state and busy-lock generation that own the user's turn. /stop marks the
// primary stopped and force-unlocks the session (bumping the generation), so
// a fallback that has not started yet must check cancelled() before it is
// created and before it is sent the message.
type fallbackGuard struct {
	primary        *interactiveState
	session        *Session
	sessions       *SessionManager
	lockGen        uint64
	interactiveKey string
	ccSessionKey   string
}

func (g fallbackGuard) cancelled(e *Engine) bool {
	if e.ctx.Err() != nil {
		return true
	}
	if g.primary != nil && g.primary.isStopped() {
		return true
	}
	return g.lockGen != 0 && !g.session.HoldsLock(g.lockGen)
}

func queuedMessageFromMessage(p Platform, msg *Message) queuedMessage {
	return queuedMessage{
		messageID:         msg.MessageID,
		platform:          p,
		replyCtx:          msg.ReplyCtx,
		content:           msg.Content,
		images:            msg.Images,
		files:             msg.Files,
		fromVoice:         msg.FromVoice,
		userID:            msg.UserID,
		userName:          msg.UserName,
		msgPlatform:       msg.Platform,
		msgSessionKey:     msg.SessionKey,
		channelKey:        msg.ChannelKey,
		userMessageTimeMs: msg.UserMessageTimeMs,
	}
}

// agentFallbackChain holds the per-project fallback agents, built lazily on
// first use, and the primary's blocked-until state.
type agentFallbackChain struct {
	specs   []AgentFallbackSpec
	factory func(name string, opts map[string]any) (Agent, error)
	now     func() time.Time

	mu       sync.Mutex
	agents   []Agent
	sessions []*SessionManager
	// handedOff records, per fallback state key and primary session, the
	// Timestamp of the newest primary history entry the fallback agent has
	// seen (History timestamps, not c.now, so the clocks always match).
	handedOff map[string]fallbackWatermark

	blockedUntil time.Time
	blockedKind  ErrorKind
}

type fallbackWatermark struct {
	sessionID string
	at        time.Time
}

func newAgentFallbackChain(specs []AgentFallbackSpec) *agentFallbackChain {
	copied := make([]AgentFallbackSpec, len(specs))
	for i, s := range specs {
		opts := make(map[string]any, len(s.Options))
		for k, v := range s.Options {
			opts[k] = v
		}
		copied[i] = AgentFallbackSpec{Type: s.Type, Options: opts, Label: s.Label}
	}
	return &agentFallbackChain{
		specs:     copied,
		factory:   CreateAgent,
		now:       time.Now,
		agents:    make([]Agent, len(specs)),
		sessions:  make([]*SessionManager, len(specs)),
		handedOff: make(map[string]fallbackWatermark),
	}
}

// SetAgentFallbacks configures the agent fallback chain. Call before Start.
// An empty list disables the feature (default).
func (e *Engine) SetAgentFallbacks(specs []AgentFallbackSpec) {
	if len(specs) == 0 {
		e.agentFallback = nil
		return
	}
	e.agentFallback = newAgentFallbackChain(specs)
}

func fallbackStateKey(interactiveKey string, i int) string {
	return fmt.Sprintf("%s#fallback:%d", interactiveKey, i)
}

func isFallbackStateKey(key string) bool {
	return strings.Contains(key, "#fallback:")
}

// externalSessionKey maps an internal fallback state key back to the user's
// conversation key ("chat#fallback:0" → "chat"). Hooks and other external
// consumers only ever see the conversation key.
func externalSessionKey(key string) string {
	if i := strings.Index(key, "#fallback:"); i >= 0 {
		return key[:i]
	}
	return key
}

// agentAt returns fallback agent i and its session manager, building them on
// first use. A build error is returned every time (no caching), so a missing
// CLI that gets installed later starts working without a restart.
func (c *agentFallbackChain) agentAt(i int) (Agent, *SessionManager, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.agents[i] != nil {
		return c.agents[i], c.sessions[i], nil
	}
	spec := c.specs[i]
	opts := make(map[string]any, len(spec.Options))
	for k, v := range spec.Options {
		opts[k] = v
	}
	a, err := c.factory(spec.Type, opts)
	if err != nil {
		return nil, nil, fmt.Errorf("fallback agent %d (%s): %w", i, spec.Type, err)
	}
	c.agents[i] = a
	// In-memory: fallback conversations are a stop-gap; after a restart the
	// fallback agent starts fresh and gets the hand-off history again.
	c.sessions[i] = NewSessionManager("")
	return a, c.sessions[i], nil
}

// label returns the notice name for fallback i.
func (c *agentFallbackChain) label(i int, a Agent) string {
	spec := c.specs[i]
	if spec.Label != "" {
		return spec.Label
	}
	name := agentDisplayName(a)
	model, _ := spec.Options["model"].(string)
	if model == "" {
		if ma, ok := a.(interface{ GetModel() string }); ok {
			model = ma.GetModel()
		}
	}
	if model != "" {
		return fmt.Sprintf("%s (%s)", name, model)
	}
	return name
}

func agentDisplayName(a Agent) string {
	if a == nil {
		return ""
	}
	if dn, ok := a.(AgentDisplayNamer); ok {
		if n := dn.DisplayName(); n != "" {
			return n
		}
	}
	if info, ok := a.(AgentDoctorInfo); ok {
		if n := info.CLIDisplayName(); n != "" {
			return n
		}
	}
	return a.Name()
}

// primaryBlocked reports whether the primary is inside a known limit window.
func (c *agentFallbackChain) primaryBlocked() (time.Time, ErrorKind, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.blockedUntil.IsZero() {
		return time.Time{}, ErrorKindUnknown, false
	}
	if !c.now().Before(c.blockedUntil) {
		c.blockedUntil = time.Time{}
		c.blockedKind = ErrorKindUnknown
		return time.Time{}, ErrorKindUnknown, false
	}
	return c.blockedUntil, c.blockedKind, true
}

// recordPrimaryFailure remembers the primary's reset time, if it gave one.
func (c *agentFallbackChain) recordPrimaryFailure(kind ErrorKind, resetAt time.Time) {
	if resetAt.IsZero() || kind == ErrorKindOverloaded {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	if !resetAt.After(now) {
		return
	}
	if limit := now.Add(maxPrimaryBlock); resetAt.After(limit) {
		resetAt = limit
	}
	c.blockedUntil = resetAt
	c.blockedKind = kind
}

// fallbackNotice builds the one-line notice sent before a fallback turn.
func (e *Engine) fallbackNotice(failed string, cause turnOutcome, next string) string {
	if cause.blocked {
		return e.i18n.Tf(MsgAgentFallbackBlocked, failed, formatResetTime(cause.ResetAt, time.Now()), next)
	}
	switch cause.Kind {
	case ErrorKindUsageLimit:
		return e.i18n.Tf(MsgAgentFallbackUsageLimit, failed, next)
	case ErrorKindRateLimit:
		return e.i18n.Tf(MsgAgentFallbackRateLimit, failed, next)
	case ErrorKindOverloaded:
		return e.i18n.Tf(MsgAgentFallbackOverloaded, failed, next)
	}
	return e.i18n.Tf(MsgAgentFallbackFailed, failed, next)
}

// formatResetTime renders a reset time in local time: "15:04 MST" today,
// "Jan 2 15:04 MST" on another day.
func formatResetTime(t, now time.Time) string {
	lt, ln := t.Local(), now.Local()
	if lt.Year() == ln.Year() && lt.YearDay() == ln.YearDay() {
		return lt.Format("15:04 MST")
	}
	return lt.Format("Jan 2 15:04 MST")
}

// answerWithFallbacks answers cause.Input with the fallback chain after the
// primary failed (or was skipped). session/sessions are the primary's.
func (e *Engine) answerWithFallbacks(guard fallbackGuard, cause turnOutcome) {
	chain := e.agentFallback
	in := cause.Input
	if !cause.blocked {
		chain.recordPrimaryFailure(cause.Kind, cause.ResetAt)
	}
	failedLabel := agentDisplayName(e.agent)
	last := len(chain.specs) - 1
	for i := range chain.specs {
		if guard.cancelled(e) {
			slog.Info("agent fallback: turn cancelled before fallback", "session_key", guard.interactiveKey, "index", i)
			return
		}
		agent, fbSessions, err := chain.agentAt(i)
		if err != nil {
			slog.Error("agent fallback: cannot build fallback agent", "index", i, "error", err)
			cause = turnOutcome{Intercepted: true, ErrText: err.Error(), Input: in}
			continue
		}
		label := chain.label(i, agent)
		slog.Warn("agent fallback: answering with fallback agent",
			"session_key", guard.interactiveKey, "index", i, "fallback", label,
			"kind", cause.Kind, "blocked", cause.blocked, "reset_at", cause.ResetAt)
		e.send(in.platform, in.replyCtx, e.fallbackNotice(failedLabel, cause, label))

		mode := interceptAnyError
		if i == last {
			mode = interceptNone
		}
		out := e.runFallbackTurn(guard, i, agent, fbSessions, in, mode)
		if !out.Intercepted {
			// Answered, failed with the error shown, or cancelled.
			return
		}
		cause = out
		cause.Input = in
		failedLabel = label
	}
	// Only reached when no fallback produced a visible result (every agent
	// failed to build or start, or the last one could not start).
	msg := cause.ErrText
	if msg == "" {
		msg = "all fallback agents failed"
	}
	e.send(in.platform, in.replyCtx, fmt.Sprintf(e.i18n.T(MsgError), msg))
}

// runFallbackTurn runs one turn of in on fallback agent i.
func (e *Engine) runFallbackTurn(guard fallbackGuard, i int, agent Agent, fbSessions *SessionManager, in queuedMessage, mode fallbackIntercept) turnOutcome {
	chain := e.agentFallback
	session, sessions := guard.session, guard.sessions
	interactiveKey, ccSessionKey := guard.interactiveKey, guard.ccSessionKey
	fbKey := fallbackStateKey(interactiveKey, i)
	// One fallback conversation per primary cc-connect session: /new on the
	// primary gives the fallback agent a fresh conversation too.
	fbSession := fbSessions.GetOrCreateActive(session.ID)

	ccKey := ccSessionKey
	if ccKey == "" {
		ccKey = interactiveKey
	}
	if guard.cancelled(e) {
		return turnOutcome{Cancelled: true}
	}
	state := e.getOrCreateInteractiveStateWith(fbKey, in.platform, in.replyCtx, fbSession, fbSessions, agent, ccKey)
	// Re-check now that the state is registered: a /stop that ran before
	// registration could not see it (stopFallbackStates found nothing), but
	// it marked the primary stopped / bumped the lock generation first.
	if guard.cancelled(e) {
		e.cleanupInteractiveState(fbKey, state)
		return turnOutcome{Cancelled: true}
	}
	if state.agentSession == nil {
		return turnOutcome{Intercepted: true, ErrText: e.i18n.T(MsgFailedToStartAgentSession), Input: in}
	}
	e.cancelAgentSessionIdleClose(state)

	state.mu.Lock()
	state.platform = in.platform
	state.replyCtx = in.replyCtx
	state.currentMessageID = in.messageID
	state.fromVoice = in.fromVoice
	state.sideText = ""
	state.currentTurnUserMessageTimeMs = in.userMessageTimeMs
	state.fallbackIntercept = mode
	state.isFallbackState = true
	state.turnInput = in
	as := state.agentSession
	state.mu.Unlock()

	// Fallback states never run an unsolicited reader; always start from a
	// clean channel (e.g. a late EventResult after an earlier failure).
	drainEvents(as.Events())

	prompt := e.buildSenderPrompt(in.content, in.userID, in.userName, in.msgPlatform, in.msgSessionKey, in.channelKey)
	prompt = chain.historyPrefix(fbKey, session, fbSession) + prompt
	fbSession.AddHistory("user", in.content)
	histBefore := fbSession.HistoryLen()

	if guard.cancelled(e) || state.isStopped() {
		e.cleanupInteractiveState(fbKey, state)
		return turnOutcome{Cancelled: true}
	}
	var stopTyping func()
	if ti, ok := in.platform.(TypingIndicator); ok {
		stopTyping = ti.StartTyping(e.ctx, in.replyCtx)
	}
	sendDone := make(chan error, 1)
	go func() {
		sendDone <- as.Send(prompt, in.messageID, in.images, in.files)
	}()
	out := e.processInteractiveEvents(state, fbSession, fbSessions, fbKey, in.messageID, time.Now(), stopTyping, sendDone, in.replyCtx, 0)

	state.mu.Lock()
	state.fallbackIntercept = interceptNone
	alive := state.agentSession != nil && state.agentSession.Alive() && !state.stopped
	state.mu.Unlock()
	if alive {
		e.scheduleAgentSessionIdleClose(fbKey, state)
	}

	if out.Intercepted {
		slog.Warn("agent fallback: fallback agent failed", "index", i, "kind", out.Kind, "error", out.ErrText)
		return out
	}
	// Mirror the fallback's answer into the primary session history.
	if hist := fbSession.GetHistory(0); len(hist) > histBefore {
		if last := hist[len(hist)-1]; last.Role == "assistant" {
			session.AddHistory("assistant", last.Content)
			sessions.Save()
		}
	}
	// Everything in the primary history up to now (including the mirrored
	// answer) is known to the fallback agent's own conversation.
	if newest := session.GetHistory(1); len(newest) == 1 {
		chain.markHandedOff(fbKey, session.ID, newest[0].Timestamp)
	}
	return out
}

func (c *agentFallbackChain) markHandedOff(fbKey, sessionID string, at time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.handedOff[fbKey] = fallbackWatermark{sessionID: sessionID, at: at}
}

// historyPrefix renders the primary conversation the fallback agent has not
// seen yet as plain text. The newest primary entry is the current user
// message (already added by the caller), so it is left out; the prompt
// carries it.
func (c *agentFallbackChain) historyPrefix(fbKey string, session *Session, fbSession *Session) string {
	c.mu.Lock()
	wm := c.handedOff[fbKey]
	c.mu.Unlock()
	// The watermark only counts while the fallback agent still holds that
	// conversation; a fresh fallback session (no agent session ID yet, or a
	// different primary session) gets the full recent context.
	var since time.Time
	if wm.sessionID == session.ID && fbSession.GetAgentSessionID() != "" {
		since = wm.at
	}

	hist := session.GetHistory(0)
	if n := len(hist); n > 0 && hist[n-1].Role == "user" {
		hist = hist[:n-1]
	}
	var picked []HistoryEntry
	for _, h := range hist {
		if h.Role != "user" && h.Role != "assistant" {
			continue
		}
		if strings.TrimSpace(h.Content) == "" || !h.Timestamp.After(since) {
			continue
		}
		picked = append(picked, h)
	}
	if len(picked) > fallbackHistoryEntries {
		picked = picked[len(picked)-fallbackHistoryEntries:]
	}
	if len(picked) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[Context: earlier messages of this conversation, answered by another assistant. Oldest first.]\n")
	for _, h := range picked {
		role := "User"
		if h.Role == "assistant" {
			role = "Assistant"
		}
		b.WriteString(role)
		b.WriteString(": ")
		b.WriteString(truncateRunes(strings.TrimSpace(h.Content), fallbackHistoryEntryMaxRunes))
		b.WriteString("\n")
	}
	b.WriteString("[End of context. The current message follows.]\n\n")
	return b.String()
}

// stopFallbackStates stops fallback turns that belong to interactiveKey, so
// /stop also halts a fallback agent that is answering for it.
func (e *Engine) stopFallbackStates(interactiveKey string, notifyQueued bool) {
	if e.agentFallback == nil || isFallbackStateKey(interactiveKey) {
		return
	}
	for i := range e.agentFallback.specs {
		e.stopInteractiveSessionWithOptions(fallbackStateKey(interactiveKey, i), notifyQueued)
	}
}

// cleanupFallbackStates closes fallback agent sessions that belong to
// interactiveKey (session reset, cron run end, idle reap of the primary).
func (e *Engine) cleanupFallbackStates(interactiveKey string) {
	if e.agentFallback == nil || isFallbackStateKey(interactiveKey) {
		return
	}
	for i := range e.agentFallback.specs {
		key := fallbackStateKey(interactiveKey, i)
		e.interactiveMu.Lock()
		_, ok := e.interactiveStates[key]
		e.interactiveMu.Unlock()
		if ok {
			e.cleanupInteractiveState(key)
		}
	}
}

// stopFallbackAgents stops the fallback agents built so far (engine Stop).
func (e *Engine) stopFallbackAgents() []error {
	if e.agentFallback == nil {
		return nil
	}
	c := e.agentFallback
	c.mu.Lock()
	agents := append([]Agent(nil), c.agents...)
	c.mu.Unlock()
	var errs []error
	for _, a := range agents {
		if a == nil {
			continue
		}
		if err := a.Stop(); err != nil {
			errs = append(errs, fmt.Errorf("stop fallback agent %s: %w", a.Name(), err))
		}
	}
	return errs
}
