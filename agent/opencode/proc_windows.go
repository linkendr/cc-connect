//go:build windows

package opencode

import (
	"os"
	"os/exec"
)

// prepareProcessGroup is a no-op on Windows.
func prepareProcessGroup(cmd *exec.Cmd) {}

// killProcessGroup kills the direct child on Windows (no process groups).
func killProcessGroup(cmd *exec.Cmd) {
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}

// killedBySIGKILL: Windows has no signals.
func killedBySIGKILL(ps *os.ProcessState) bool { return false }

// leaderExited: not needed on Windows (no signals); unknown.
func leaderExited(pid int) (exited, known bool) { return false, false }
