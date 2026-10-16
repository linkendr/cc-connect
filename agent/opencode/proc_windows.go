//go:build windows

package opencode

import "os/exec"

// prepareProcessGroup is a no-op on Windows; cancellation kills the direct
// child (exec.CommandContext's default).
func prepareProcessGroup(cmd *exec.Cmd) {}
