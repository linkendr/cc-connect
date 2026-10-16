package core

import (
	"testing"
	"time"
)

func TestClassifyAnthropicError(t *testing.T) {
	cases := []struct {
		name    string
		payload string
		want    ErrorKind
	}{
		// Canonical Anthropic error JSON (from upstream PR #557).
		{"shape A rate limit", `{"error":{"type":"rate_limit_error","message":"rate limit exceeded"}}`, ErrorKindRateLimit},
		{"shape A overloaded", `{"error":{"type":"overloaded_error","message":"overloaded"}}`, ErrorKindOverloaded},
		{"shape B top-level type", `{"type":"rate_limit_error"}`, ErrorKindRateLimit},
		{"embedded JSON with prefix", `API Error: {"error":{"type":"rate_limit_error","message":"slow down"}}`, ErrorKindRateLimit},
		{"embedded JSON with shell-style prefix", `claude: error: {"error":{"type":"overloaded_error"}}`, ErrorKindOverloaded},
		{"literal fallback rate_limit_error in prose", `Request failed: rate_limit_error: too many requests from this key`, ErrorKindRateLimit},
		{"literal fallback overloaded_error in prose", `Upstream returned overloaded_error after 3 retries`, ErrorKindOverloaded},
		{"literal fallback case-insensitive", `API RETURNED Rate_Limit_Error STATUS`, ErrorKindRateLimit},
		{"malformed JSON with canonical token", `{not valid json but contains rate_limit_error somewhere}`, ErrorKindRateLimit},
		{"API Error 529 wrapper", `API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, ErrorKindOverloaded},

		// Claude subscription usage limits (Claude Code 2.1.x wording).
		{"session limit with reset", "You've hit your session limit · resets 3pm (America/Los_Angeles)", ErrorKindUsageLimit},
		{"weekly limit with date reset", "You've hit your weekly limit · resets Oct 20, 9am (America/Los_Angeles)", ErrorKindUsageLimit},
		{"opus limit curly apostrophe", "You’ve hit your Opus limit · resets 11pm", ErrorKindUsageLimit},
		{"spend limit", "You've hit your monthly spend limit.", ErrorKindUsageLimit},
		{"team budget", "You've hit your team's shared budget. /model to switch", ErrorKindUsageLimit},
		{"reached model limit", "You've reached your Fable limit.", ErrorKindUsageLimit},
		{"out of usage credits", "You're out of usage credits · resets 5pm", ErrorKindUsageLimit},
		{"out of extra usage", "You're out of extra usage", ErrorKindUsageLimit},
		{"legacy pipe format", "Claude AI usage limit reached|1760000000", ErrorKindUsageLimit},
		{"generic usage limit resets", "Usage limit exceeded for this plan; resets at 14:00", ErrorKindUsageLimit},
		{"usage limit wins over rate_limit_error json", `API Error: 429 {"error":{"type":"rate_limit_error","message":"You've hit your session limit"}}`, ErrorKindUsageLimit},

		// Not retriable.
		{"invalid_request_error not retried", `{"error":{"type":"invalid_request_error","message":"bad model"}}`, ErrorKindUnknown},
		{"authentication_error not retried", `{"error":{"type":"authentication_error","message":"bad key"}}`, ErrorKindUnknown},
		{"api_error (500) not retried by this helper", `{"error":{"type":"api_error","message":"server fail"}}`, ErrorKindUnknown},
		{"compilation error — unrelated prose", `internal compiler error at line 42`, ErrorKindUnknown},
		{"generic rate limit prose without canonical token", `You are being rate limited, please wait`, ErrorKindUnknown},
		{"HTTP 429 text alone", `HTTP 429 Too Many Requests`, ErrorKindUnknown},
		{"max turns", `Reached max turns (10)`, ErrorKindUnknown},
		{"usage limit without reset", `the usage limit field must be positive`, ErrorKindUnknown},
		{"hit your head, not a limit", `You've hit your head on the wall.`, ErrorKindUnknown},
		{"empty", "", ErrorKindUnknown},
		{"nested error object without type", `{"error":{"message":"something went wrong"}}`, ErrorKindUnknown},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyAnthropicError(tc.payload)
			if got != tc.want {
				t.Errorf("ClassifyAnthropicError(%q) = %q, want %q", tc.payload, got, tc.want)
			}
		})
	}
}

func TestErrorKindIsRetriable(t *testing.T) {
	cases := []struct {
		kind ErrorKind
		want bool
	}{
		{ErrorKindUnknown, false},
		{ErrorKindRateLimit, true},
		{ErrorKindOverloaded, true},
		{ErrorKindUsageLimit, true},
		{ErrorKind("auth"), false},
	}
	for _, tc := range cases {
		if got := tc.kind.IsRetriable(); got != tc.want {
			t.Errorf("(%q).IsRetriable() = %v, want %v", tc.kind, got, tc.want)
		}
	}
}

func TestParseLimitResetTime(t *testing.T) {
	la, err := time.LoadLocation("America/Los_Angeles")
	if err != nil {
		t.Skipf("tzdata unavailable: %v", err)
	}
	// 2026-10-16 10:00 in Los Angeles.
	now := time.Date(2026, 10, 16, 10, 0, 0, 0, la)

	cases := []struct {
		name string
		text string
		want time.Time
	}{
		{"pipe unix seconds", "Claude AI usage limit reached|1792200000", time.Unix(1792200000, 0)},
		{"pipe unix in the past is ignored", "Claude AI usage limit reached|1000000000", time.Time{}},
		{"retry-after header", "429 rate_limit_error retry-after: 30", now.Add(30 * time.Second)},
		{"retry after seconds", "please retry after 120 seconds", now.Add(120 * time.Second)},
		{"clock later today", "You've hit your session limit · resets 3pm (America/Los_Angeles)", time.Date(2026, 10, 16, 15, 0, 0, 0, la)},
		{"clock with minutes", "You've hit your session limit · resets 3:30pm (America/Los_Angeles)", time.Date(2026, 10, 16, 15, 30, 0, 0, la)},
		{"clock already passed rolls to tomorrow", "You've hit your session limit · resets 9am (America/Los_Angeles)", time.Date(2026, 10, 17, 9, 0, 0, 0, la)},
		{"12am is midnight", "resets 12am (America/Los_Angeles)", time.Date(2026, 10, 17, 0, 0, 0, 0, la)},
		{"24h clock no zone uses now's zone", "usage limit; resets at 14:00", time.Date(2026, 10, 16, 14, 0, 0, 0, la)},
		{"other zone", "resets 6pm (UTC)", time.Date(2026, 10, 16, 18, 0, 0, 0, time.UTC)},
		{"date form", "You've hit your weekly limit · resets Oct 20, 9am (America/Los_Angeles)", time.Date(2026, 10, 20, 9, 0, 0, 0, la)},
		{"date form past rolls to next year", "resets Jan 2, 9am (America/Los_Angeles)", time.Date(2027, 1, 2, 9, 0, 0, 0, la)},
		{"no reset info", "You've hit your monthly spend limit.", time.Time{}},
		{"relative form unsupported", "resets in 2h 10m", time.Time{}},
		{"invalid clock", "resets 27pm", time.Time{}},
		{"empty", "", time.Time{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseLimitResetTime(tc.text, now)
			if !got.Equal(tc.want) {
				t.Errorf("ParseLimitResetTime(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}
