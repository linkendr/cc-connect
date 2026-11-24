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
