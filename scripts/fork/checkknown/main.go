// Command checkknown is the fork's merge gate for upstream syncs.
//
// It runs `go test ./... -race -json` three times (or reads saved -json
// output files given as arguments) and compares the failing tests with
// scripts/fork/known-upstream-failures.txt. Rules are in that file's header.
// Exit 0: no new failure and no stale entry. Exit 1: either. Exit 2: usage or
// run error.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
)

const (
	defaultList = "scripts/fork/known-upstream-failures.txt"
	failQuorum  = 2 // a test counts as failing when it fails in at least this many runs
)

// testEvent is the subset of a test2json event this check reads.
type testEvent struct {
	Action      string `json:"Action"`
	Package     string `json:"Package"`
	Test        string `json:"Test"`
	FailedBuild string `json:"FailedBuild"`
}

const (
	pkgSuffix        = " PKG"
	unfinishedSuffix = " (unfinished)"
)

// isStructural reports whether id marks a broken run (cut-short stream, crash,
// build failure) instead of an ordinary test failure. One such id in any run
// is a problem; ordinary failures need failQuorum runs.
func isStructural(id string) bool {
	return strings.HasSuffix(id, pkgSuffix) || strings.HasSuffix(id, unfinishedSuffix)
}

// failureSet maps "<package> <test>" (or "<package> PKG") to a flag.
type failureSet map[string]bool

// parseRun reads one `go test -json` stream and returns the failing ids.
//
//   - a test with a fail event is "<package> <test>";
//   - a test that started and never finished (crash, os.Exit, kill) is also failing;
//   - a package that failed to build, never reported a result (stream cut
//     short), or failed with no failing test is "<package> PKG".
func parseRun(r io.Reader) (failureSet, error) {
	failed := failureSet{}
	pkgSeen := map[string]bool{}
	pkgDone := map[string]bool{}
	pkgFailed := map[string]bool{}
	pkgHasTestFail := map[string]bool{}
	running := map[string]bool{} // "<package> <test>" started, no result yet
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	events := 0
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 || line[0] != '{' {
			continue // non-JSON noise (build output)
		}
		var ev testEvent
		if err := json.Unmarshal(line, &ev); err != nil {
			continue
		}
		events++
		if ev.Package == "" {
			continue
		}
		pkgSeen[ev.Package] = true
		id := ev.Package + " " + ev.Test
		switch ev.Action {
		case "run":
			if ev.Test != "" {
				running[id] = true
			}
		case "pass", "skip", "fail":
			if ev.Test == "" {
				pkgDone[ev.Package] = true
				if ev.Action == "fail" {
					pkgFailed[ev.Package] = true
				}
				continue
			}
			delete(running, id)
			if ev.Action == "fail" {
				pkgHasTestFail[ev.Package] = true
				failed[id] = true
			}
		}
		if ev.FailedBuild != "" {
			pkgFailed[ev.Package] = true
			pkgHasTestFail[ev.Package] = false
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if events == 0 {
		return nil, errors.New("no test2json events in input")
	}
	for id := range running {
		failed[id+unfinishedSuffix] = true
		pkg, _, _ := strings.Cut(id, " ")
		pkgHasTestFail[pkg] = true
	}
	for p := range pkgSeen {
		if !pkgDone[p] || (pkgFailed[p] && !pkgHasTestFail[p]) {
			failed[p+" PKG"] = true
		}
	}
	return failed, nil
}

// parseList reads the known-failures file into a set of ids.
func parseList(r io.Reader) (failureSet, error) {
	known := failureSet{}
	sc := bufio.NewScanner(r)
	n := 0
	for sc.Scan() {
		n++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 3 {
			return nil, fmt.Errorf("line %d: want <package> <test> <upstream commit>, got %q", n, line)
		}
		known[f[0]+" "+f[1]] = true
	}
	return known, sc.Err()
}

// evaluate applies the rules to the runs and returns the problems found.
func evaluate(known failureSet, runs []failureSet) (newFailures, stale []string) {
	counts := map[string]int{}
	for _, r := range runs {
		for id := range r {
			counts[id]++
		}
	}
	for id, c := range counts {
		if isStructural(id) {
			newFailures = append(newFailures, fmt.Sprintf("%s (broken run, seen in %d of %d runs)", id, c, len(runs)))
			continue
		}
		if c >= failQuorum && !known[id] {
			newFailures = append(newFailures, fmt.Sprintf("%s (failed %d of %d runs)", id, c, len(runs)))
		}
	}
	for id := range known {
		if counts[id] == 0 {
			stale = append(stale, id)
		}
	}
	sort.Strings(newFailures)
	sort.Strings(stale)
	return newFailures, stale
}

// runGoTest runs the suite once. It cross-checks go test's exit code with the
// stream: exit 0 with a failing id, or exit 1 with no failing id, is an error.
func runGoTest() (failureSet, error) {
	cmd := exec.Command("go", "test", "./...", "-race", "-json")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr
	code := 0
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			return nil, err
		}
		code = ee.ExitCode()
	}
	set, err := parseRun(&out)
	if err != nil {
		return nil, err
	}
	switch {
	case code > 1 || code < 0:
		return nil, fmt.Errorf("go test exited with code %d (not a test failure)", code)
	case code == 0 && len(set) > 0:
		return nil, fmt.Errorf("go test exited 0 but the stream has %d failing ids", len(set))
	case code == 1 && len(set) == 0:
		return nil, errors.New("go test exited 1 but the stream has no failing id")
	}
	return set, nil
}

// checkWebDist fails early when the web assets that web/embed.go embeds are missing.
func checkWebDist() error {
	entries, err := os.ReadDir("web/dist")
	if err != nil || len(entries) == 0 {
		return errors.New("web/dist is missing or empty: run `pnpm install --frozen-lockfile && pnpm build` in web/ with pnpm 10 first")
	}
	return nil
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("checkknown", flag.ContinueOnError)
	fs.SetOutput(stderr)
	list := fs.String("list", defaultList, "known-failures file")
	runsN := fs.Int("runs", 3, "go test runs when no JSON files are given")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	lf, err := os.Open(*list)
	if err != nil {
		fmt.Fprintln(stderr, "checkknown:", err)
		return 2
	}
	known, err := parseList(lf)
	lf.Close()
	if err != nil {
		fmt.Fprintln(stderr, "checkknown:", err)
		return 2
	}
	var runs []failureSet
	if fs.NArg() > 0 {
		for _, p := range fs.Args() {
			f, err := os.Open(p)
			if err != nil {
				fmt.Fprintln(stderr, "checkknown:", err)
				return 2
			}
			r, err := parseRun(f)
			f.Close()
			if err != nil {
				fmt.Fprintf(stderr, "checkknown: %s: %v\n", p, err)
				return 2
			}
			runs = append(runs, r)
		}
	} else {
		if err := checkWebDist(); err != nil {
			fmt.Fprintln(stderr, "checkknown:", err)
			return 2
		}
		for i := 0; i < *runsN; i++ {
			r, err := runGoTest()
			if err != nil {
				fmt.Fprintf(stderr, "checkknown: run %d: %v\n", i+1, err)
				return 2
			}
			fmt.Fprintf(stdout, "run %d: %d failing ids\n", i+1, len(r))
			runs = append(runs, r)
		}
	}
	newF, stale := evaluate(known, runs)
	for _, s := range newF {
		fmt.Fprintln(stdout, "NEW FAILURE:", s)
	}
	for _, s := range stale {
		fmt.Fprintln(stdout, "STALE ENTRY (passes in every run, delete it from the list):", s)
	}
	if len(newF)+len(stale) > 0 {
		fmt.Fprintf(stdout, "checkknown: FAIL (%d new, %d stale)\n", len(newF), len(stale))
		return 1
	}
	fmt.Fprintf(stdout, "checkknown: OK (%d runs, %d known upstream failures)\n", len(runs), len(known))
	return 0
}
