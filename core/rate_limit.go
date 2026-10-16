package core

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ClassifyAnthropicError inspects an Anthropic-style error payload and
// returns a structured ErrorKind. It accepts these input shapes:
//
//  1. A full JSON object like `{"error":{"type":"rate_limit_error",...}}`.
//  2. A JSON object with a top-level `"type"` field.
//  3. A wrapper string that embeds (1) or (2) after some prefix text,
//     e.g. `API Error: {"error":{"type":"rate_limit_error"}}`.
//  4. A Claude subscription usage-limit message, e.g.
//     "You've hit your session limit · resets 3pm (America/Los_Angeles)" or
//     the older "Claude AI usage limit reached|1760000000".
//
// Detection of (1)-(3) targets Anthropic's canonical `error.type` strings
// ("rate_limit_error", "overloaded_error") — these come from the API's typed
// error schema, not free-form prose. (4) matches the fixed phrases Claude
// Code prints when a subscription cap is hit. Callers must only pass text
// that is already known to be an error (stderr of a failed process, or the
// result of an is_error turn): an ordinary answer may quote these phrases.
// Returns ErrorKindUnknown when nothing matches, in which case callers
// should NOT fall back or retry.
//
// Adapted from upstream PR #557 (not merged); the usage-limit shape is new.
func ClassifyAnthropicError(payload string) ErrorKind {
	if payload == "" {
		return ErrorKindUnknown
	}

	if IsUsageLimitText(payload) {
		return ErrorKindUsageLimit
	}

	// 1. Try parsing the whole payload as JSON.
	if k := classifyJSON([]byte(payload)); k != ErrorKindUnknown {
		return k
	}

	// 2. If the payload has a prefix before an embedded JSON object, slice
	//    from the first '{' and retry. CLI stderr often looks like
	//    "Error: {"error":{"type":"rate_limit_error"}}".
	if i := strings.Index(payload, "{"); i > 0 {
		if k := classifyJSON([]byte(payload[i:])); k != ErrorKindUnknown {
			return k
		}
	}

	// 3. Last-resort literal fallback: match Anthropic's canonical error-type
	//    tokens directly. We still require the full token — not partial
	//    words like "rate limit" — so we don't misclassify ordinary prose.
	low := strings.ToLower(payload)
	if strings.Contains(low, "rate_limit_error") {
		return ErrorKindRateLimit
	}
	if strings.Contains(low, "overloaded_error") {
		return ErrorKindOverloaded
	}
	return ErrorKindUnknown
}

// classifyJSON walks an Anthropic error JSON object and extracts error.type.
func classifyJSON(b []byte) ErrorKind {
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		return ErrorKindUnknown
	}
	// Shape A: {"error": {"type": "..."}}
	if inner, ok := raw["error"].(map[string]any); ok {
		if t, _ := inner["type"].(string); t != "" {
			return kindFromAnthropicType(t)
		}
	}
	// Shape B: {"type": "..."}
	if t, _ := raw["type"].(string); t != "" {
		return kindFromAnthropicType(t)
	}
	return ErrorKindUnknown
}

// kindFromAnthropicType maps Anthropic's error.type strings to ErrorKind.
// Unknown types return ErrorKindUnknown so callers don't fall back on auth
// or validation failures.
func kindFromAnthropicType(t string) ErrorKind {
	switch t {
	case "rate_limit_error":
		return ErrorKindRateLimit
	case "overloaded_error":
		return ErrorKindOverloaded
	}
	return ErrorKindUnknown
}

// IsRetriable reports whether an ErrorKind means "the backend is temporarily
// unable to serve this turn" — the engine may answer it with a fallback agent.
func (k ErrorKind) IsRetriable() bool {
	return k == ErrorKindRateLimit || k == ErrorKindOverloaded || k == ErrorKindUsageLimit
}

// usageLimitPhrases are the fixed, lower-cased phrases Claude Code uses when
// a subscription usage cap stops a turn. Checked against Claude Code 2.1.301:
// the message is built as "You've hit your <limit name><suffix>" (limit names
// "session limit", "weekly limit", "Opus limit", "Sonnet limit", "Fable
// limit", "usage credit limit", "monthly spend limit"), plus "You've reached
// your <model> limit.", "You're out of usage credits" and "You're out of
// extra usage". Older CLIs printed "Claude AI usage limit reached|<unix>".
var usageLimitPhrases = []string{
	"usage limit reached",
	"you're out of usage credits",
	"you're out of extra usage",
}

// usageLimitHitRe matches "You've hit/reached your ... limit|budget" with
// either apostrophe style, limited to one short clause so a long error
// paragraph that merely mentions a limit later is not matched.
var usageLimitHitRe = regexp.MustCompile(`(?i)you(?:'|’|\s+ha)ve\s+(?:hit|reached)\s+your\s+[^.\n]{0,60}?(?:limit|budget)`)

// IsUsageLimitText reports whether text is a Claude subscription usage-limit
// message. Only meaningful for text already known to be an error.
func IsUsageLimitText(text string) bool {
	if text == "" {
		return false
	}
	low := strings.ToLower(strings.ReplaceAll(text, "’", "'"))
	for _, p := range usageLimitPhrases {
		if strings.Contains(low, p) {
			return true
		}
	}
	if usageLimitHitRe.MatchString(text) {
		return true
	}
	// Generic "usage limit ... resets ..." wording (e.g. a provider proxy
	// rephrasing the cap) — require both halves to avoid matching prose.
	if strings.Contains(low, "usage limit") && strings.Contains(low, "reset") {
		return true
	}
	return false
}

var (
	// "Claude AI usage limit reached|1760000000" (older Claude Code).
	usageLimitPipeRe = regexp.MustCompile(`(?i)limit reached\|(\d{9,12})`)
	// "retry-after: 30", "retry after 30s", "Retry-After=30".
	retryAfterRe = regexp.MustCompile(`(?i)retry[-_ ]after["']?\s*[:=]?\s*(\d{1,6})\s*(?:s|sec|secs|seconds)?\b`)
	// "resets 3pm", "resets at 3:30pm", "resets 15:00", optional "(Area/City)".
	resetsClockRe = regexp.MustCompile(`(?i)resets\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([A-Za-z_]+/[A-Za-z_/+-]+|UTC)\))?`)
	// "resets Oct 20, 9am (Area/City)" or "resets Oct 20 at 9:30am".
	resetsDateRe = regexp.MustCompile(`(?i)resets\s+(?:on\s+)?([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([A-Za-z_]+/[A-Za-z_/+-]+|UTC)\))?`)
)

// ParseLimitResetTime extracts when a rate/usage limit clears from an error
// message, relative to now. It understands:
//
//   - "usage limit reached|<unix seconds>" (older Claude Code)
//   - "retry-after: N" / "retry after N seconds"
//   - "resets 3pm (America/Los_Angeles)" / "resets at 15:30" — next
//     occurrence of that wall-clock time in the named zone (now's zone when
//     absent)
//   - "resets Oct 20, 9am (America/Los_Angeles)"
//
// Returns the zero time when no reset time is present or it cannot be
// parsed; callers then treat the block as unknown and keep trying the
// primary agent on every new message.
func ParseLimitResetTime(text string, now time.Time) time.Time {
	if text == "" {
		return time.Time{}
	}
	if m := usageLimitPipeRe.FindStringSubmatch(text); m != nil {
		if sec, err := strconv.ParseInt(m[1], 10, 64); err == nil {
			if t := time.Unix(sec, 0); t.After(now) {
				return t
			}
		}
	}
	if m := retryAfterRe.FindStringSubmatch(text); m != nil {
		if sec, err := strconv.Atoi(m[1]); err == nil && sec > 0 {
			return now.Add(time.Duration(sec) * time.Second)
		}
	}
	if m := resetsDateRe.FindStringSubmatch(text); m != nil {
		if t := resolveResetDate(m[1], m[2], m[3], m[4], m[5], m[6], now); !t.IsZero() {
			return t
		}
	}
	if m := resetsClockRe.FindStringSubmatch(text); m != nil {
		if t := resolveResetClock(m[1], m[2], m[3], m[4], now); !t.IsZero() {
			return t
		}
	}
	return time.Time{}
}

func resetLocation(tz string, now time.Time) *time.Location {
	if tz == "" {
		return now.Location()
	}
	loc, err := time.LoadLocation(tz)
	if err != nil {
		return now.Location()
	}
	return loc
}

// clock24 converts hour/minute/am-pm captures to a 24-hour clock.
func clock24(hourS, minS, ampm string) (int, int, bool) {
	hour, err := strconv.Atoi(hourS)
	if err != nil {
		return 0, 0, false
	}
	minute := 0
	if minS != "" {
		if minute, err = strconv.Atoi(minS); err != nil || minute > 59 {
			return 0, 0, false
		}
	}
	switch strings.ToLower(ampm) {
	case "am":
		if hour < 1 || hour > 12 {
			return 0, 0, false
		}
		if hour == 12 {
			hour = 0
		}
	case "pm":
		if hour < 1 || hour > 12 {
			return 0, 0, false
		}
		if hour != 12 {
			hour += 12
		}
	default:
		if hour > 23 {
			return 0, 0, false
		}
	}
	return hour, minute, true
}

func resolveResetClock(hourS, minS, ampm, tz string, now time.Time) time.Time {
	hour, minute, ok := clock24(hourS, minS, ampm)
	if !ok {
		return time.Time{}
	}
	loc := resetLocation(tz, now)
	local := now.In(loc)
	t := time.Date(local.Year(), local.Month(), local.Day(), hour, minute, 0, 0, loc)
	if !t.After(now) {
		t = t.AddDate(0, 0, 1)
	}
	return t
}

func resolveResetDate(monS, dayS, hourS, minS, ampm, tz string, now time.Time) time.Time {
	mon, err := time.Parse("Jan", strings.ToUpper(monS[:1])+strings.ToLower(monS[1:3]))
	if err != nil {
		return time.Time{}
	}
	day, err := strconv.Atoi(dayS)
	if err != nil || day < 1 || day > 31 {
		return time.Time{}
	}
	hour, minute, ok := clock24(hourS, minS, ampm)
	if !ok {
		return time.Time{}
	}
	loc := resetLocation(tz, now)
	local := now.In(loc)
	t := time.Date(local.Year(), mon.Month(), day, hour, minute, 0, 0, loc)
	if !t.After(now) {
		t = t.AddDate(1, 0, 0)
	}
	return t
}

// IsTerminal reports whether an ErrorKind is user-caused or terminal (max
// turns, max budget, cancelled, context too long): another agent would hit
// the same limit, or the user stopped the turn.
func (k ErrorKind) IsTerminal() bool {
	switch k {
	case ErrorKindMaxTurns, ErrorKindMaxBudget, ErrorKindCancelled, ErrorKindContextTooLong:
		return true
	}
	return false
}
