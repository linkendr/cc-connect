//go:build unix && !darwin && !linux

package opencode

// leaderExited cannot read process state on this platform; callers treat
// unknown as "still running".
func leaderExited(pid int) (exited, known bool) { return false, false }
