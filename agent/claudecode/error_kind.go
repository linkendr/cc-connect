package claudecode

import (
	"strings"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

// This file classifies failed Claude Code turns so the engine can answer the
// same message with a fallback agent. It only reads stream-json output and
// stderr; it does not change how the claude CLI is started or talks to
// Anthropic.
//
// Output shapes (Agent SDK reference, checked against Claude Code 2.1.301):
//
//   - result, subtype "success", is_error true, api_error_status 429,
//     result "You've hit your session limit · resets 3pm (America/Los_Angeles)"
//     — the subscription usage cap.
//   - result, subtype "error_during_execution" (or another error_* subtype),
//     no `result` field, messages in `errors: []string`.
//   - assistant, `error` one of "rate_limit", "overloaded", "billing_error", …
//     on the message that wraps an API error.
//   - rate_limit_event, rate_limit_info {status "allowed"|"allowed_warning"|
//     "rejected", resetsAt <unix seconds>}.

// nonRetriableResultSubtypes are turn failures caused by the caller's own
// limits, not by the backend. Another agent would hit the same limit or the
// user must change something, so they never trigger a fallback.
var nonRetriableResultSubtypes = map[string]bool{
	"error_max_turns":                     true,
	"error_max_budget_usd":                true,
	"error_max_structured_output_retries": true,
}

// resultErrorText returns the human-readable error text of a result event:
// `result` for the "success"-subtype API-error shape, else the joined
// `errors` array of the error_* subtypes.
func resultErrorText(raw map[string]any) string {
	if s, ok := raw["result"].(string); ok && strings.TrimSpace(s) != "" {
		return s
	}
	if arr, ok := raw["errors"].([]any); ok {
		parts := make([]string, 0, len(arr))
		for _, v := range arr {
			if s, ok := v.(string); ok && s != "" {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "; ")
	}
	return ""
}

// classifyResultError reports whether a result event ended the turn with a
// retriable backend error and, if known, when it clears.
//
// assistantErr is the `error` field of the last assistant message of the turn
// ("" if none). rejectedResetAt is resetsAt (unix seconds) of a "rejected"
// rate_limit_event seen during the turn (0 if none).
func classifyResultError(raw map[string]any, assistantErr string, rejectedResetAt int64, now time.Time) (core.ErrorKind, time.Time) {
	subtype := resultSubtype(raw)
	isErr, _ := raw["is_error"].(bool)
	if !isErr && (subtype == "" || subtype == "success") {
		return core.ErrorKindUnknown, time.Time{}
	}
	if nonRetriableResultSubtypes[subtype] {
		// Terminal for an agent fallback chain: another agent would hit the
		// same caller-side limit.
		switch subtype {
		case "error_max_turns":
			return core.ErrorKindMaxTurns, time.Time{}
		case "error_max_budget_usd":
			return core.ErrorKindMaxBudget, time.Time{}
		}
		return core.ErrorKindUnknown, time.Time{}
	}

	text := resultErrorText(raw)
	kind := core.ClassifyAnthropicError(text)
	if kind == core.ErrorKindUnknown {
		if status, ok := raw["api_error_status"].(float64); ok {
			switch int(status) {
			case 429:
				kind = core.ErrorKindRateLimit
			case 529:
				kind = core.ErrorKindOverloaded
			}
		}
	}
	if kind == core.ErrorKindUnknown {
		switch assistantErr {
		case "rate_limit":
			kind = core.ErrorKindRateLimit
		case "overloaded":
			kind = core.ErrorKindOverloaded
		case "billing_error":
			kind = core.ErrorKindUsageLimit
		}
	}
	// A 429 during a turn whose unified (subscription) limiter reported
	// "rejected" is the usage cap, not a short per-minute throttle.
	if kind == core.ErrorKindRateLimit && rejectedResetAt != 0 {
		kind = core.ErrorKindUsageLimit
	}
	if !kind.IsRetriable() {
		return core.ErrorKindUnknown, time.Time{}
	}
	return kind, limitResetAt(kind, text, rejectedResetAt, now)
}

// classifyStderrError classifies the stderr of a claude process that exited
// with an error (PR #557's stderr path).
func classifyStderrError(stderr string, rejectedResetAt int64, now time.Time) (core.ErrorKind, time.Time) {
	kind := core.ClassifyAnthropicError(stderr)
	if kind == core.ErrorKindRateLimit && rejectedResetAt != 0 {
		kind = core.ErrorKindUsageLimit
	}
	if !kind.IsRetriable() {
		return core.ErrorKindUnknown, time.Time{}
	}
	return kind, limitResetAt(kind, stderr, rejectedResetAt, now)
}

// limitResetAt picks the reset time: the structured rate_limit_event value
// first, then whatever the message text says. Overloaded (529) is transient
// and never blocks the primary for later messages.
func limitResetAt(kind core.ErrorKind, text string, rejectedResetAt int64, now time.Time) time.Time {
	if kind == core.ErrorKindOverloaded {
		return time.Time{}
	}
	if rejectedResetAt > 0 {
		if t := time.Unix(rejectedResetAt, 0); t.After(now) {
			return t
		}
	}
	return core.ParseLimitResetTime(text, now)
}

// parseRateLimitEvent returns resetsAt (unix seconds) when a rate_limit_event
// reports status "rejected", 0 when it reports the limit is open again, and
// ok=false when the event carries no usable status.
func parseRateLimitEvent(raw map[string]any) (resetAt int64, ok bool) {
	info, _ := raw["rate_limit_info"].(map[string]any)
	if info == nil {
		return 0, false
	}
	status, _ := info["status"].(string)
	switch status {
	case "rejected":
		if v, ok := info["resetsAt"].(float64); ok && v > 0 {
			return int64(v), true
		}
		// Rejected without a reset time: remember "rejected" with an
		// unknown reset (-1 keeps it non-zero for the usage-limit upgrade).
		return -1, true
	case "allowed", "allowed_warning":
		return 0, true
	}
	return 0, false
}
