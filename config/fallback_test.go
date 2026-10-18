package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/BurntSushi/toml"
)

// TestAgentFallback_DecodeOwnerShape decodes the documented config shape and
// checks the convenience Model maps into the "model" option.
func TestAgentFallback_DecodeOwnerShape(t *testing.T) {
	raw := `
[[projects]]
name = "home"
[projects.agent]
type = "claudecode"
fallback = [
  { type = "opencode", model = "openai/gpt-6-sol" },
  { type = "opencode", model = "deepseek/deepseek-v4-pro", label = "DeepSeek", options = { mode = "yolo" } },
]
[[projects.platforms]]
type = "discord"
[projects.platforms.options]
token = "x"
`
	var cfg Config
	if _, err := toml.Decode(raw, &cfg); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	fb := cfg.Projects[0].Agent.Fallback
	if len(fb) != 2 {
		t.Fatalf("fallback len = %d, want 2", len(fb))
	}
	if fb[0].Type != "opencode" || fb[0].Model != "openai/gpt-6-sol" {
		t.Errorf("fallback[0] = %+v", fb[0])
	}
	opts := fb[1].EffectiveOptions()
	if opts["model"] != "deepseek/deepseek-v4-pro" || opts["mode"] != "yolo" {
		t.Errorf("fallback[1] options = %#v", opts)
	}
	if fb[1].Label != "DeepSeek" {
		t.Errorf("fallback[1].Label = %q", fb[1].Label)
	}
	// EffectiveOptions must not alias the config map.
	opts["mode"] = "changed"
	if fb[1].Options["mode"] != "yolo" {
		t.Error("EffectiveOptions aliased the config Options map")
	}
}

func TestAgentFallback_Validate(t *testing.T) {
	type fb = FallbackAgentConfig
	cases := []struct {
		name         string
		primaryModel string
		mode         string
		runAsUser    string
		runAsEnv     []string
		primaryOpts  map[string]any
		chain        []FallbackAgentConfig
		wantErr      string // "" = valid
	}{
		{name: "empty chain is valid", chain: nil},
		{name: "owner example", chain: []fb{{Type: "opencode", Model: "openai/gpt-6-sol"}, {Type: "opencode", Model: "deepseek/deepseek-v4-pro"}}},
		{name: "model via options", chain: []fb{{Type: "opencode", Options: map[string]any{"model": "openai/gpt-6-sol"}}}},
		{name: "same type as primary with other model", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet"}}},
		{name: "three entries allowed", chain: []fb{{Type: "opencode", Model: "a"}, {Type: "opencode", Model: "b"}, {Type: "codex"}}},
		{name: "four entries rejected", chain: []fb{{Type: "opencode", Model: "a"}, {Type: "opencode", Model: "b"}, {Type: "codex"}, {Type: "gemini"}}, wantErr: "at most 3"},
		{name: "missing type", chain: []fb{{Model: "x"}}, wantErr: "fallback[0].type is required"},
		{name: "spaced type", chain: []fb{{Type: " opencode"}}, wantErr: "surrounding spaces"},
		{name: "self reference no model", chain: []fb{{Type: "claudecode"}}, wantErr: "is the primary agent"},
		{name: "self reference same model", primaryModel: "opus", chain: []fb{{Type: "opencode", Model: "x"}, {Type: "claudecode", Model: "opus"}}, wantErr: "fallback[1] (claudecode, model \"opus\") is the primary agent"},
		{name: "duplicate entry", chain: []fb{{Type: "opencode", Model: "x"}, {Type: "opencode", Options: map[string]any{"model": "x"}}}, wantErr: "repeats fallback[0]"},
		{name: "model conflict", chain: []fb{{Type: "opencode", Model: "x", Options: map[string]any{"model": "y"}}}, wantErr: "use one"},
		{name: "non-string options.model", chain: []fb{{Type: "opencode", Options: map[string]any{"model": 3}}}, wantErr: "options.model must be a string"},
		{name: "nested fallback", chain: []fb{{Type: "opencode", Options: map[string]any{"fallback": []any{}}}}, wantErr: "do not nest"},
		{name: "multi-workspace rejected", mode: "multi-workspace", chain: []fb{{Type: "opencode"}}, wantErr: "not supported"},
		{name: "non-string work_dir", chain: []fb{{Type: "opencode", Options: map[string]any{"work_dir": int64(42)}}}, wantErr: "options.work_dir must be a string, got int64"},
		{name: "non-string mode", chain: []fb{{Type: "opencode", Options: map[string]any{"mode": true}}}, wantErr: "options.mode must be a string"},
		{name: "run_as_user project cannot fall back to opencode", runAsUser: "agent", chain: []fb{{Type: "opencode", Model: "x"}}, wantErr: `type "opencode" does not support run_as_user/run_as_env`},
		{name: "run_as_env project cannot fall back to opencode", runAsEnv: []string{"FOO"}, chain: []fb{{Type: "opencode"}}, wantErr: "does not support run_as_user"},
		{name: "run_as_user in fallback options for opencode", chain: []fb{{Type: "opencode", Options: map[string]any{"run_as_user": "agent"}}}, wantErr: "does not support run_as_user"},
		{name: "run_as_user project may fall back to claudecode", runAsUser: "agent", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet"}}},
		{name: "integer run_as_user in primary options", primaryOpts: map[string]any{"run_as_user": int64(7)}, chain: []fb{{Type: "claudecode", Model: "sonnet"}}, wantErr: "agent.options.run_as_user must be a string, got int64"},
		{name: "integer run_as_user in claudecode fallback", runAsUser: "agent", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet", Options: map[string]any{"run_as_user": int64(42)}}}, wantErr: "fallback[0].options.run_as_user must be a string, got int64"},
		{name: "string run_as_env in claudecode fallback", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet", Options: map[string]any{"run_as_env": "FOO"}}}, wantErr: "fallback[0].options.run_as_env must be a list of strings, got string"},
		{name: "non-string run_as_env item", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet", Options: map[string]any{"run_as_env": []any{"FOO", int64(1)}}}}, wantErr: "run_as_env[1] must be a string, got int64"},
		{name: "dangerous run_as_env in fallback", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet", Options: map[string]any{"run_as_env": []any{"LD_PRELOAD"}}}}, wantErr: "dangerous variable"},
		{name: "credit leg: same type and model as primary with its own env", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]any{"ANTHROPIC_API_KEY": "test-credit-value"}}}, {Type: "opencode", Model: "x"}}},
		{name: "credit leg: primary env differs from fallback env", primaryModel: "opus", primaryOpts: map[string]any{"env": map[string]any{"FOO": "a"}}, chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]any{"FOO": "b"}}}}},
		{name: "same model and same env as primary is the primary", primaryModel: "opus", primaryOpts: map[string]any{"env": map[string]any{"FOO": "a"}}, chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]string{"FOO": "a"}}}}, wantErr: "is the primary agent"},
		{name: "primary env is not inherited: env-less same-model fallback differs", primaryModel: "opus", primaryOpts: map[string]any{"env": map[string]any{"FOO": "a"}}, chain: []fb{{Type: "claudecode", Model: "opus"}}},
		{name: "two credit legs with the same env repeat", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]any{"ANTHROPIC_API_KEY": "k"}}}, {Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]any{"ANTHROPIC_API_KEY": "k"}}}}, wantErr: "repeats fallback[0]"},
		{name: "credit leg with blank env value (unset placeholder)", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]any{"ANTHROPIC_API_KEY": ""}}}}, wantErr: "only by empty options.env values"},
		{name: "non-string env value", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": map[string]any{"ANTHROPIC_API_KEY": int64(1)}}}}, wantErr: "options.env [ANTHROPIC_API_KEY] must be a string"},
		{name: "non-table env", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": "ANTHROPIC_API_KEY=x"}}}, wantErr: "options.env must be a table of strings"},
		{name: "valid run_as options in claudecode fallback", primaryModel: "opus", chain: []fb{{Type: "claudecode", Model: "sonnet", Options: map[string]any{"run_as_user": "agent", "run_as_env": []any{"FOO"}}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			proj := validProject("demo")
			proj.Agent.Type = "claudecode"
			if tc.primaryModel != "" {
				proj.Agent.Options = map[string]any{"model": tc.primaryModel}
			}
			proj.Mode = tc.mode
			proj.RunAsUser = tc.runAsUser
			for k, v := range tc.primaryOpts {
				proj.Agent.Options[k] = v
			}
			proj.RunAsEnv = tc.runAsEnv
			proj.Agent.Fallback = tc.chain
			err := validateAgentFallback("projects[0]", proj)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("validateAgentFallback = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("validateAgentFallback = %v, want error containing %q", err, tc.wantErr)
			}
		})
	}
}

// TestAgentFallback_ValidateWiredIntoLoad proves Config.validate runs the
// fallback check (not only the helper).
func TestAgentFallback_ValidateWiredIntoLoad(t *testing.T) {
	cfg := Config{Projects: []ProjectConfig{validProject("demo")}}
	cfg.Projects[0].Agent.Fallback = []FallbackAgentConfig{{Type: ""}}
	err := cfg.validate()
	if err == nil || !strings.Contains(err.Error(), "projects[0].agent.fallback[0].type is required") {
		t.Fatalf("validate() = %v, want fallback type error", err)
	}
}

// label and model are typed string fields: a non-string value fails at decode.
func TestAgentFallback_DecodeRejectsNonStringFields(t *testing.T) {
	for _, entry := range []string{`{ type = "opencode", label = 42 }`, `{ type = "opencode", model = 7 }`} {
		raw := "[[projects]]\nname = \"p\"\n[projects.agent]\ntype = \"claudecode\"\nfallback = [ " + entry + " ]\n"
		var cfg Config
		if _, err := toml.Decode(raw, &cfg); err == nil {
			t.Errorf("decode of %s succeeded, want a type error", entry)
		}
	}
}

// TestAgentFallback_CreditLegErrorsNeverShowEnvValues: validation errors
// about env-distinguished fallbacks name keys at most, never values.
func TestAgentFallback_CreditLegErrorsNeverShowEnvValues(t *testing.T) {
	const secret = "sk-ant-test-secret-value"
	env := map[string]any{"ANTHROPIC_API_KEY": secret}
	for _, chain := range [][]FallbackAgentConfig{
		{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": env}}, {Type: "claudecode", Model: "opus", Options: map[string]any{"env": env}}},
		{{Type: "claudecode", Model: "opus", Options: map[string]any{"env": env}}},
	} {
		proj := validProject("demo")
		proj.Agent.Type = "claudecode"
		proj.Agent.Options = map[string]any{"model": "opus", "env": env}
		proj.Agent.Fallback = chain
		err := validateAgentFallback("projects[0]", proj)
		if err == nil {
			t.Fatal("validateAgentFallback = nil, want a duplicate error")
		}
		if strings.Contains(err.Error(), secret) {
			t.Fatalf("error leaks the env value: %v", err)
		}
	}
}

// TestAgentFallback_CreditLegEnvPlaceholder loads the documented credit-leg
// shape from a file: ${VAR} in options.env resolves from the process
// environment, and an unset variable is rejected instead of silently
// starting a second copy of the primary.
func TestAgentFallback_CreditLegEnvPlaceholder(t *testing.T) {
	raw := `
[[projects]]
name = "home"
[projects.agent]
type = "claudecode"
fallback = [
  { type = "claudecode", model = "opus", label = "Claude Code (API credit)", options = { env = { ANTHROPIC_API_KEY = "${CC_TEST_CREDIT_KEY}" } } },
  { type = "opencode", model = "openai/gpt-6-sol" },
]
[projects.agent.options]
model = "opus"
[[projects.platforms]]
type = "discord"
[projects.platforms.options]
token = "x"
`
	path := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("CC_TEST_CREDIT_KEY", "test-credit-value")
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	fb := cfg.Projects[0].Agent.Fallback
	if len(fb) != 2 || fb[0].Label != "Claude Code (API credit)" {
		t.Fatalf("fallback = %+v", fb)
	}
	env, _ := fb[0].EffectiveOptions()["env"].(map[string]any)
	if env["ANTHROPIC_API_KEY"] != "test-credit-value" {
		t.Errorf("env placeholder not resolved: got %d entries", len(env))
	}
	if _, ok := cfg.Projects[0].Agent.Options["env"]; ok {
		t.Error("primary options gained an env table")
	}

	if err := os.Unsetenv("CC_TEST_CREDIT_KEY"); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "only by empty options.env values") {
		t.Fatalf("Load with unset placeholder = %v, want empty-env error", err)
	}
}
