package core

import (
	"errors"
	"strings"
)

// Fork-only (linkendr/cc-connect): Engine fields and helpers that upstream
// does not have. They live here, not in engine.go, so the daily upstream
// merge does not conflict on them; engine.go embeds engineFork in Engine.

// engineFork holds the fork's Engine fields.
type engineFork struct {
	// menuSkills limits the skills in the platform command menu. nil = every
	// skill; else only these names. Guarded by Engine.userRolesMu.
	menuSkills map[string]bool
	// agentFallback answers turns the primary agent cannot serve (rate/usage
	// limit, overloaded) with other agents. nil = disabled. See
	// agent_fallback.go.
	agentFallback *agentFallbackChain
}

// SetMenuSkills limits the skills registered in the platform command menu to
// names. A nil slice keeps every skill. Skills left out of the menu can still
// be run by typing /name.
func (e *Engine) SetMenuSkills(names []string) {
	e.userRolesMu.Lock()
	defer e.userRolesMu.Unlock()
	if names == nil {
		e.menuSkills = nil
		return
	}
	e.menuSkills = make(map[string]bool, len(names))
	for _, n := range names {
		e.menuSkills[strings.ToLower(strings.TrimPrefix(n, "/"))] = true
	}
}

// ErrAgentExitedWithoutOutput is wrapped by an agent adapter's EventError when
// the agent process failed without saying anything (the adapter keeps the exit
// detail in the wrapped text for logs). The engine shows the user the
// translated MsgAgentExitedWithoutOutput instead of the English text.
var ErrAgentExitedWithoutOutput = errors.New("agent exited without output")

// ErrAgentCloseTimeout is wrapped by an agent adapter's Close error when the
// agent's reader did not stop in time. The engine shows the user the translated
// MsgAgentCloseTimeout; the adapter keeps the detail for logs.
var ErrAgentCloseTimeout = errors.New("agent close timed out")

// ErrAgentProcessStart is wrapped by an agent adapter's Send error when it could
// not set up or start the agent process (pipes, exec). The engine shows the
// user the translated MsgAgentProcessStartFailed; the OS detail stays in logs.
var ErrAgentProcessStart = errors.New("agent process could not be started")

// agentErrorText is the text of an agent error for the user: a known
// generated diagnostic is translated, any other error keeps its own text.
func (e *Engine) agentErrorText(err error) string {
	if err == nil {
		return ""
	}
	if errors.Is(err, ErrAgentProcessStart) {
		return e.i18n.T(MsgAgentProcessStartFailed)
	}
	if errors.Is(err, ErrAgentExitedWithoutOutput) {
		return e.i18n.T(MsgAgentExitedWithoutOutput)
	}
	if errors.Is(err, ErrAgentCloseTimeout) {
		return e.i18n.T(MsgAgentCloseTimeout)
	}
	return err.Error()
}
