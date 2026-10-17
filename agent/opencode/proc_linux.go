//go:build linux

package opencode

import (
	"os"
	"strconv"
	"strings"
)

// leaderExited reports whether pid has exited (a zombie, not yet reaped).
// known=false when the state cannot be read.
func leaderExited(pid int) (exited, known bool) {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false, false
	}
	// Format: pid (comm) state ...; comm may contain spaces or ')'.
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 || i+2 >= len(s) {
		return false, false
	}
	return s[i+2] == 'Z' || s[i+2] == 'X', true
}
