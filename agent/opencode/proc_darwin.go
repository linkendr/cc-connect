//go:build darwin

package opencode

import "golang.org/x/sys/unix"

// leaderExited reports whether pid has exited (a zombie, not yet reaped).
// known=false when the state cannot be read.
func leaderExited(pid int) (exited, known bool) {
	kp, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil || kp == nil {
		return false, false
	}
	const szomb = 5 // SZOMB in <sys/proc.h>
	return kp.Proc.P_stat == szomb, true
}
