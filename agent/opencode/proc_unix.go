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

// processGroupID is the child's process group (its PID, via Setpgid).
func processGroupID(cmd *exec.Cmd) int {
	if cmd == nil || cmd.Process == nil {
		return 0
	}
	return cmd.Process.Pid
}

// killGroup SIGKILLs process group pgid; a gone group is not an error.
func killGroup(pgid int) {
	if pgid > 0 {
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
	}
}

// processGroupAlive reports whether process group pgid still has members.
func processGroupAlive(pgid int) bool {
	err := syscall.Kill(-pgid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
