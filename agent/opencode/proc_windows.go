//go:build windows

package opencode

import "os/exec"

// prepareProcessGroup is a no-op on Windows; cancellation kills the direct
// child (exec.CommandContext's default).
func prepareProcessGroup(cmd *exec.Cmd) {}

// Windows has no process groups here: the session kills the direct child.
func processGroupID(cmd *exec.Cmd) int { return 0 }

func killGroup(pgid int) {}

func processGroupAlive(pgid int) bool { return false }
