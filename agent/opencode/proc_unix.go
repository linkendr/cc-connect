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
}

// killProcessGroup SIGKILLs the child's whole process group (the child
// alone if the group signal fails for another reason than "already gone").
// Callers only use it while the child is unreaped (opencodeProc).
func killProcessGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil &&
		!errors.Is(err, os.ErrProcessDone) && !errors.Is(err, syscall.ESRCH) {
		_ = cmd.Process.Kill()
	}
}

// killedBySIGKILL reports whether the process was ended by SIGKILL (the
// turn-end kill of a CLI that had not exited yet).
func killedBySIGKILL(ps *os.ProcessState) bool {
	ws, ok := ps.Sys().(syscall.WaitStatus)
	return ok && ws.Signaled() && ws.Signal() == syscall.SIGKILL
}
