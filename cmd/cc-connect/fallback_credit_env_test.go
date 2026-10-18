//go:build !windows && !no_claudecode

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/config"
	"github.com/chenhg5/cc-connect/core"
)

// TestFallbackCreditLeg_EnvReachesOnlyTheFallbackChild builds the credit leg
// (same claude CLI and model as the primary, plus ANTHROPIC_API_KEY in its
// options.env) the way main.go does, starts both agents against a fake
// claude, and reads what each child process actually got in its env.
func TestFallbackCreditLeg_EnvReachesOnlyTheFallbackChild(t *testing.T) {
	// The supervisor's own ANTHROPIC_API_KEY would reach both children;
	// unset it so the test sees only what cc-connect adds.
	if old, ok := os.LookupEnv("ANTHROPIC_API_KEY"); ok {
		if err := os.Unsetenv("ANTHROPIC_API_KEY"); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Setenv("ANTHROPIC_API_KEY", old) })
	}

	bin := filepath.Join(t.TempDir(), "fake-claude")
	// Record the key (a dummy test value) in the work dir, announce a
	// session, then stay alive until stdin closes.
	script := "#!/bin/sh\n" +
		"printf '%s' \"${ANTHROPIC_API_KEY-<unset>}\" > \"$PWD/child-env.tmp\"\n" +
		"mv \"$PWD/child-env.tmp\" \"$PWD/child-env\"\n" +
		"echo '{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"fake\"}'\n" +
		"cat > /dev/null\n"
	if err := os.WriteFile(bin, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	primaryDir, fallbackDir := t.TempDir(), t.TempDir()
	proj := config.ProjectConfig{
		Name: "home",
		Agent: config.AgentConfig{
			Type:    "claudecode",
			Options: map[string]any{"model": "opus", "cmd": bin, "work_dir": primaryDir},
			Fallback: []config.FallbackAgentConfig{{
				Type:  "claudecode",
				Model: "opus",
				Options: map[string]any{
					"cmd":      bin,
					"work_dir": fallbackDir,
					"env":      map[string]any{"ANTHROPIC_API_KEY": "test-credit-value"},
				},
			}},
		},
	}
	dataDir := t.TempDir()
	specs, err := buildAgentFallbackSpecs(dataDir, proj, []string{"claudecode"})
	if err != nil {
		t.Fatalf("buildAgentFallbackSpecs: %v", err)
	}
	primaryOpts := buildAgentOptions(dataDir, proj)
	if _, ok := primaryOpts["env"]; ok {
		t.Fatal("primary options carry an env table")
	}

	start := func(opts map[string]any, dir string) string {
		t.Helper()
		agent, err := core.CreateAgent("claudecode", opts)
		if err != nil {
			t.Fatalf("CreateAgent: %v", err)
		}
		sess, err := agent.StartSession(context.Background(), "")
		if err != nil {
			t.Fatalf("StartSession: %v", err)
		}
		t.Cleanup(func() { _ = sess.Close() })
		out := filepath.Join(dir, "child-env")
		// Generous: under a loaded `go test ./...` the spawn can take seconds.
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			if b, err := os.ReadFile(out); err == nil {
				return string(b)
			}
			time.Sleep(10 * time.Millisecond)
		}
		t.Fatalf("fake claude never wrote %s", out)
		return ""
	}

	if got := start(specs[0].Options, fallbackDir); got != "test-credit-value" {
		t.Errorf("fallback child ANTHROPIC_API_KEY = %q, want the configured test value", got)
	}
	if got := start(primaryOpts, primaryDir); got != "<unset>" {
		t.Errorf("primary child ANTHROPIC_API_KEY = %q, want <unset>", got)
	}
}
