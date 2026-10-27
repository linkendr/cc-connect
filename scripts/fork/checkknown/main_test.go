package main

import (
	"strings"
	"testing"
)

func stream(lines ...string) string { return strings.Join(lines, "\n") + "\n" }

func TestParseRunTestAndPackageFailures(t *testing.T) {
	in := stream(
		`{"Action":"fail","Package":"p/a","Test":"TestX"}`,
		`{"Action":"fail","Package":"p/a"}`,
		`{"Action":"fail","Package":"p/b"}`,
		`{"Action":"pass","Package":"p/c","Test":"TestOK"}`,
		`{"Action":"pass","Package":"p/c"}`,
		`not json`,
	)
	got, err := parseRun(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if !got["p/a TestX"] || !got["p/b PKG"] || got["p/a PKG"] || len(got) != 2 {
		t.Fatalf("unexpected set %v", got)
	}
}

func TestParseRunIncompleteStreams(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"known test failed then package never finished",
			stream(`{"Action":"run","Package":"p/a","Test":"TestKnown"}`, `{"Action":"fail","Package":"p/a","Test":"TestKnown"}`),
			[]string{"p/a PKG", "p/a TestKnown"}},
		{"test started and never finished",
			stream(`{"Action":"run","Package":"p/a","Test":"TestCrash"}`, `{"Action":"fail","Package":"p/a"}`),
			[]string{"p/a TestCrash (unfinished)"}},
		{"build failure",
			stream(`{"Action":"fail","Package":"p/a","FailedBuild":"p/a"}`),
			[]string{"p/a PKG"}},
		{"clean pass",
			stream(`{"Action":"run","Package":"p/a","Test":"TestOK"}`, `{"Action":"pass","Package":"p/a","Test":"TestOK"}`, `{"Action":"pass","Package":"p/a"}`),
			nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseRun(strings.NewReader(tc.in))
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
			for _, w := range tc.want {
				if !got[w] {
					t.Fatalf("got %v want %v", got, tc.want)
				}
			}
		})
	}
}

func TestParseListRejectsBrokenRunId(t *testing.T) {
	if _, err := parseList(strings.NewReader("p/a PKG 140c1821\n")); err == nil {
		t.Fatal("want error for a PKG entry")
	}
}

func TestParseRunEmptyIsError(t *testing.T) {
	if _, err := parseRun(strings.NewReader("garbage\n")); err == nil {
		t.Fatal("want error for stream without events")
	}
}

func TestParseListRejectsMalformedLine(t *testing.T) {
	if _, err := parseList(strings.NewReader("p/a TestX\n")); err == nil {
		t.Fatal("want error for a line without commit")
	}
	k, err := parseList(strings.NewReader("# c\n\np/a TestX 140c1821\nflaky p/a TestY 140c1821\n"))
	if err != nil || !k.failing["p/a TestX"] || !k.flaky["p/a TestY"] {
		t.Fatalf("got %v %v", k, err)
	}
}

func TestEvaluate(t *testing.T) {
	known := knownList{failing: failureSet{"p/a TestKnown": true, "p/a TestFixed": true}, flaky: failureSet{"p/a TestFlaky": true}}
	r := func(ids ...string) failureSet {
		s := failureSet{}
		for _, id := range ids {
			s[id] = true
		}
		return s
	}
	tests := []struct {
		name      string
		runs      []failureSet
		wantNew   int
		wantStale int
	}{
		{"only known", []failureSet{r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed")}, 0, 0},
		{"new failure twice", []failureSet{r("p/a TestKnown", "p/a TestFixed", "p/a TestNew"), r("p/a TestKnown", "p/a TestFixed", "p/a TestNew"), r("p/a TestKnown", "p/a TestFixed")}, 1, 0},
		{"new failure once is still new", []failureSet{r("p/a TestKnown", "p/a TestFixed", "p/a TestNew"), r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed")}, 1, 0},
		{"listed flaky test failing once is accepted", []failureSet{r("p/a TestKnown", "p/a TestFixed", "p/a TestFlaky"), r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed")}, 0, 0},
		{"listed flaky test passing everywhere is accepted", []failureSet{r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed")}, 0, 0},
		{"listed test now passes", []failureSet{r("p/a TestKnown"), r("p/a TestKnown"), r("p/a TestKnown")}, 0, 1},
		{"listed test failing once is kept", []failureSet{r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown"), r("p/a TestKnown")}, 0, 0},
		{"broken run once is enough", []failureSet{r("p/a TestKnown", "p/a TestFixed", "p/z PKG"), r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed")}, 1, 0},
		{"unfinished test once is enough", []failureSet{r("p/a TestKnown", "p/a TestFixed", "p/a TestCrash (unfinished)"), r("p/a TestKnown", "p/a TestFixed"), r("p/a TestKnown", "p/a TestFixed")}, 1, 0},
		{"package failure twice", []failureSet{r("p/a TestKnown", "p/a TestFixed", "p/z PKG"), r("p/a TestKnown", "p/a TestFixed", "p/z PKG"), r("p/a TestKnown", "p/a TestFixed")}, 1, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n, s := evaluate(known, tc.runs)
			if len(n) != tc.wantNew || len(s) != tc.wantStale {
				t.Fatalf("new=%v stale=%v", n, s)
			}
		})
	}
}
