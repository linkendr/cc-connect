package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/chenhg5/cc-connect/config"
)

func TestBuildAgentFallbackSpecs(t *testing.T) {
	proj := config.ProjectConfig{
		Name: "home",
		Agent: config.AgentConfig{
			Type: "claudecode",
			Options: map[string]any{
				"work_dir":    "/repo",
				"mode":        "bypassPermissions",
				"run_as_user": "agent",
				"model":       "opus", // not inherited
			},
			Fallback: []config.FallbackAgentConfig{
				{Type: "opencode", Model: "openai/gpt-6-sol"},
				{Type: "opencode", Model: "deepseek/deepseek-v4-pro", Label: "DeepSeek", Options: map[string]any{"mode": "default", "work_dir": "/other"}},
			},
		},
	}
	specs, err := buildAgentFallbackSpecs("/data", proj, []string{"claudecode", "opencode"})
	if err != nil {
		t.Fatalf("buildAgentFallbackSpecs: %v", err)
	}
	if len(specs) != 2 {
		t.Fatalf("len = %d, want 2", len(specs))
	}
	want0 := map[string]any{
		"model": "openai/gpt-6-sol", "work_dir": "/repo", "mode": "bypassPermissions",
		"run_as_user": "agent", "cc_data_dir": "/data", "cc_project": "home",
	}
	if specs[0].Type != "opencode" || !reflect.DeepEqual(specs[0].Options, want0) {
		t.Errorf("specs[0] = %+v, want options %#v", specs[0], want0)
	}
	if specs[1].Options["mode"] != "default" || specs[1].Options["work_dir"] != "/other" || specs[1].Label != "DeepSeek" {
		t.Errorf("specs[1] did not keep its own options: %+v", specs[1])
	}
	// The primary's options are not modified.
	if _, ok := proj.Agent.Options["cc_project"]; ok {
		t.Error("primary options were mutated")
	}
}

func TestBuildAgentFallbackSpecs_UnknownType(t *testing.T) {
	proj := config.ProjectConfig{Name: "home", Agent: config.AgentConfig{
		Type:     "claudecode",
		Fallback: []config.FallbackAgentConfig{{Type: "opencodex"}},
	}}
	_, err := buildAgentFallbackSpecs("/data", proj, []string{"opencode", "claudecode"})
	if err == nil || !strings.Contains(err.Error(), `unknown agent type "opencodex"`) {
		t.Fatalf("err = %v, want unknown agent type", err)
	}
}

func TestBuildAgentFallbackSpecs_EmptyChain(t *testing.T) {
	specs, err := buildAgentFallbackSpecs("/data", config.ProjectConfig{Name: "p", Agent: config.AgentConfig{Type: "claudecode"}}, nil)
	if err != nil || specs != nil {
		t.Fatalf("got (%v, %v), want (nil, nil)", specs, err)
	}
}
