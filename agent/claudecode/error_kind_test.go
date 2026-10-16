package claudecode

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

func newClassifyTestSession(t *testing.T) *claudeSession {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	cs := &claudeSession{
		events: make(chan core.Event, 8),
		ctx:    ctx,
		// Point the transcript lookup at a missing file so handleResult does
		// not read the real ~/.claude tree.
		transcriptOverride: t.TempDir() + "/none.jsonl",
	}
	cs.sessionID.Store("s1")
	cs.alive.Store(true)
	return cs
}

func nextEvent(t *testing.T, cs *claudeSession) core.Event {
	t.Helper()
	select {
	case evt := <-cs.events:
		return evt
	case <-time.After(2 * time.Second):
		t.Fatal("no event emitted")
		return core.Event{}
	}
}

// TestHandleResult_ErrorKind feeds result events in the shapes Claude Code
// emits (Agent SDK SDKResultMessage) and checks the classification carried on
// the EventResult. The event type and content never change, so a project
// without a fallback chain shows exactly what it showed before.
func TestHandleResult_ErrorKind(t *testing.T) {
	future := time.Now().Add(3 * time.Hour).Unix()
	cases := []struct {
		name         string
		pre          []map[string]any // lines fed before the result
		result       map[string]any
		wantKind     core.ErrorKind
		wantResetSet bool
	}{
		{
			name:     "success",
			result:   map[string]any{"type": "result", "subtype": "success", "is_error": false, "result": "all good"},
			wantKind: core.ErrorKindUnknown,
		},
		{
			name:     "success text quoting a limit phrase is not an error",
			result:   map[string]any{"type": "result", "subtype": "success", "is_error": false, "result": "You've hit your session limit is what Claude says"},
			wantKind: core.ErrorKindUnknown,
		},
		{
			name: "subscription usage limit with rate_limit_event reset",
			pre: []map[string]any{
				{"type": "rate_limit_event", "rate_limit_info": map[string]any{"status": "rejected", "resetsAt": float64(future), "rateLimitType": "five_hour"}},
				{"type": "assistant", "error": "rate_limit", "message": map[string]any{"content": []any{map[string]any{"type": "text", "text": "You've hit your session limit · resets 3pm"}}}},
			},
			result:       map[string]any{"type": "result", "subtype": "success", "is_error": true, "api_error_status": float64(429), "result": "You've hit your session limit · resets 3pm (America/Los_Angeles)"},
			wantKind:     core.ErrorKindUsageLimit,
			wantResetSet: true,
		},
		{
			name:         "usage limit text alone",
			result:       map[string]any{"type": "result", "subtype": "success", "is_error": true, "result": "You've hit your weekly limit · resets Oct 20, 9am (America/Los_Angeles)"},
			wantKind:     core.ErrorKindUsageLimit,
			wantResetSet: true,
		},
		{
			name:     "error_during_execution with limit message in errors[]",
			result:   map[string]any{"type": "result", "subtype": "error_during_execution", "is_error": true, "errors": []any{"API Error: 429", "You've hit your session limit"}},
			wantKind: core.ErrorKindUsageLimit,
		},
		{
			name:     "error_during_execution with rate_limit_error JSON",
			result:   map[string]any{"type": "result", "subtype": "error_during_execution", "is_error": true, "result": `{"error":{"type":"rate_limit_error","message":"rate limit exceeded"}}`},
			wantKind: core.ErrorKindRateLimit,
		},
		{
			name:     "overloaded 529",
			result:   map[string]any{"type": "result", "subtype": "error_during_execution", "is_error": true, "result": `{"error":{"type":"overloaded_error","message":"overloaded"}}`},
			wantKind: core.ErrorKindOverloaded,
		},
		{
			name:     "api_error_status 529 without text",
			result:   map[string]any{"type": "result", "subtype": "success", "is_error": true, "api_error_status": float64(529), "result": "API Error"},
			wantKind: core.ErrorKindOverloaded,
		},
		{
			name: "assistant billing_error",
			pre: []map[string]any{
				{"type": "assistant", "error": "billing_error", "message": map[string]any{"content": []any{}}},
			},
			result:   map[string]any{"type": "result", "subtype": "success", "is_error": true, "result": "Credit balance is too low"},
			wantKind: core.ErrorKindUsageLimit,
		},
		{
			name: "429 plus rejected limiter without reset upgrades to usage limit",
			pre: []map[string]any{
				{"type": "rate_limit_event", "rate_limit_info": map[string]any{"status": "rejected"}},
			},
			result:   map[string]any{"type": "result", "subtype": "success", "is_error": true, "api_error_status": float64(429), "result": "API Error: 429"},
			wantKind: core.ErrorKindUsageLimit,
		},
		{
			name:     "error_max_turns is terminal, not retriable, even with a limit word",
			result:   map[string]any{"type": "result", "subtype": "error_max_turns", "is_error": true, "errors": []any{"Reached max turns; usage limit resets never"}},
			wantKind: core.ErrorKindMaxTurns,
		},
		{
			name:     "error_max_budget_usd is terminal, not retriable",
			result:   map[string]any{"type": "result", "subtype": "error_max_budget_usd", "is_error": true},
			wantKind: core.ErrorKindMaxBudget,
		},
		{
			name:     "auth error is not retriable",
			result:   map[string]any{"type": "result", "subtype": "success", "is_error": true, "api_error_status": float64(401), "result": `{"error":{"type":"authentication_error"}}`},
			wantKind: core.ErrorKindUnknown,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs := newClassifyTestSession(t)
			for _, raw := range tc.pre {
				switch raw["type"] {
				case "assistant":
					cs.handleAssistant(raw)
				case "rate_limit_event":
					if resetAt, ok := parseRateLimitEvent(raw); ok {
						cs.rejectedResetAt.Store(resetAt)
					}
				}
			}
			// Drain any text events emitted by handleAssistant.
		drain:
			for {
				select {
				case <-cs.events:
				default:
					break drain
				}
			}
			cs.handleResult(tc.result)
			evt := nextEvent(t, cs)
			if evt.Type != core.EventResult {
				t.Fatalf("Type = %v, want EventResult (classification must not change the event type)", evt.Type)
			}
			if want, _ := tc.result["result"].(string); evt.Content != want {
				t.Errorf("Content = %q, want unchanged %q", evt.Content, want)
			}
			if evt.ErrorKind != tc.wantKind {
				t.Errorf("ErrorKind = %q, want %q", evt.ErrorKind, tc.wantKind)
			}
			if got := !evt.ErrorResetAt.IsZero(); got != tc.wantResetSet {
				t.Errorf("ErrorResetAt set = %v (%v), want %v", got, evt.ErrorResetAt, tc.wantResetSet)
			}
			// The per-turn state is consumed by the result.
			if v, _ := cs.lastAssistantErr.Load().(string); v != "" {
				t.Errorf("lastAssistantErr not cleared: %q", v)
			}
			if cs.rejectedResetAt.Load() != 0 {
				t.Errorf("rejectedResetAt not cleared: %d", cs.rejectedResetAt.Load())
			}
		})
	}
}

func TestHandleResult_RateLimitEventResetIsUsed(t *testing.T) {
	cs := newClassifyTestSession(t)
	reset := time.Now().Add(90 * time.Minute).Truncate(time.Second)
	cs.handleReadLoopLine(`{"type":"rate_limit_event","rate_limit_info":{"status":"rejected","resetsAt":` +
		strconv.FormatInt(reset.Unix(), 10) + `,"rateLimitType":"five_hour"},"uuid":"u","session_id":"s1"}`)
	cs.handleReadLoopLine(`{"type":"result","subtype":"success","is_error":true,"api_error_status":429,"result":"You've hit your session limit · resets 3pm","session_id":"s1"}`)
	evt := nextEvent(t, cs)
	if evt.ErrorKind != core.ErrorKindUsageLimit {
		t.Fatalf("ErrorKind = %q, want usage_limit", evt.ErrorKind)
	}
	if !evt.ErrorResetAt.Equal(reset) {
		t.Errorf("ErrorResetAt = %v, want rate_limit_event resetsAt %v", evt.ErrorResetAt, reset)
	}
}

func TestParseRateLimitEvent(t *testing.T) {
	cases := []struct {
		name   string
		raw    map[string]any
		want   int64
		wantOK bool
	}{
		{"rejected with reset", map[string]any{"rate_limit_info": map[string]any{"status": "rejected", "resetsAt": float64(1792200000)}}, 1792200000, true},
		{"rejected without reset", map[string]any{"rate_limit_info": map[string]any{"status": "rejected"}}, -1, true},
		{"allowed clears", map[string]any{"rate_limit_info": map[string]any{"status": "allowed", "resetsAt": float64(1792200000)}}, 0, true},
		{"allowed_warning clears", map[string]any{"rate_limit_info": map[string]any{"status": "allowed_warning"}}, 0, true},
		{"missing info", map[string]any{}, 0, false},
		{"unknown status", map[string]any{"rate_limit_info": map[string]any{"status": "weird"}}, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := parseRateLimitEvent(tc.raw)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("parseRateLimitEvent = (%d, %v), want (%d, %v)", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestClassifyStderrError(t *testing.T) {
	now := time.Now()
	cases := []struct {
		name     string
		stderr   string
		rejected int64
		want     core.ErrorKind
	}{
		{"rate limit json", `API Error: {"error":{"type":"rate_limit_error"}}`, 0, core.ErrorKindRateLimit},
		{"rate limit plus rejected limiter", `API Error: {"error":{"type":"rate_limit_error"}}`, -1, core.ErrorKindUsageLimit},
		{"overloaded", `API Error: 529 {"type":"error","error":{"type":"overloaded_error"}}`, 0, core.ErrorKindOverloaded},
		{"usage limit text", "Claude AI usage limit reached|1792200000", 0, core.ErrorKindUsageLimit},
		{"unrelated crash", "panic: runtime error: index out of range", 0, core.ErrorKindUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := classifyStderrError(tc.stderr, tc.rejected, now)
			if got != tc.want {
				t.Errorf("classifyStderrError(%q) = %q, want %q", tc.stderr, got, tc.want)
			}
		})
	}
}
