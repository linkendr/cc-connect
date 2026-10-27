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
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
}

// failureSet maps "<package> <test>" (or "<package> PKG") to a flag.
type failureSet map[string]bool

// parseRun reads one `go test -json` stream and returns the failing test ids.
// A package-level failure with no failing test in that package is recorded as
// "<package> PKG".
func parseRun(r io.Reader) (failureSet, error) {
	failed := failureSet{}
	pkgFailed := map[string]bool{}
	pkgHasTestFail := map[string]bool{}
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
		if ev.Action != "fail" || ev.Package == "" {
			continue
		}
		if ev.Test == "" {
			pkgFailed[ev.Package] = true
			continue
		}
		pkgHasTestFail[ev.Package] = true
		failed[ev.Package+" "+ev.Test] = true
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if events == 0 {
		return nil, errors.New("no test2json events in input")
	}
	for p := range pkgFailed {
		if !pkgHasTestFail[p] {
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

func runGoTest() (failureSet, error) {
	cmd := exec.Command("go", "test", "./...", "-race", "-json")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr
	// go test exits 1 when tests fail; only an unparseable stream is an error.
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			return nil, err
		}
	}
	return parseRun(&out)
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
