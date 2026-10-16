//go:build unix

package opencode

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

// prepareProcessGroup starts the `opencode run` child in its own process
// group, so cancelling the turn can kill every descendant (tool commands,
// helper processes) instead of orphaning the ones that hold its pipes open.
// Same pattern as agent/claudecode and agent/codex.
func prepareProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	cmd.Cancel = func() error { return killProcessGroup(cmd) }
}

// killProcessGroup SIGKILLs the child's whole process group.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil &&
		!errors.Is(err, os.ErrProcessDone) && !errors.Is(err, syscall.ESRCH) {
		return cmd.Process.Kill()
	}
	return nil
}
