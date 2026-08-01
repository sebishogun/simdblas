package simdblas

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The README states the minimum lengths as numbers in a sentence, and the guide
// and CHANGELOG restate them. Those are the same facts written in four places,
// which is exactly the shape that drifts: someone measures a better threshold,
// changes the constant, and the prose keeps the old one indefinitely because
// nothing fails.
//
// So the prose is checked against the constants instead.
func TestReadmeThresholdsMatchConstants(t *testing.T) {
	readme := readFile(t, "README.md")

	// "dot 64, axpy 48, scal 48, swap 48, rot 48, asum 16, nrm2 8"
	stated := map[string]int{}
	// The list wraps across lines in the README, so the separator has to allow
	// a newline as well as a comma.
	re := regexp.MustCompile(`\*\*((?:\w+ \d+[,\s]*)+)\*\*`)
	sep := regexp.MustCompile(`,\s*`)
	for _, m := range re.FindAllStringSubmatch(readme, -1) {
		for _, pair := range sep.Split(strings.TrimSpace(m[1]), -1) {
			f := strings.Fields(pair)
			if len(f) != 2 {
				continue
			}
			n, err := strconv.Atoi(f[1])
			if err != nil {
				continue
			}
			stated[f[0]] = n
		}
	}

	want := map[string]int{
		"dot": minLenDot, "axpy": minLenAxpy, "scal": minLenScal,
		"swap": minLenSwap, "rot": minLenRot, "asum": minLenAsum, "nrm2": minLenNrm2,
	}
	for name, v := range want {
		got, ok := stated[name]
		if !ok {
			t.Errorf("the README does not state a minimum length for %s; it is %d", name, v)
			continue
		}
		if got != v {
			t.Errorf("the README says the %s threshold is %d; the constant is %d", name, got, v)
		}
	}
	if len(stated) == 0 {
		t.Fatal("no thresholds parsed from the README; the sentence has been " +
			"reworded and this check no longer verifies anything")
	}
}

// Every file the README and guide point at must exist. A dead link in the one
// document a new reader starts from is worse than no document.
func TestDocLinksResolve(t *testing.T) {
	link := regexp.MustCompile(`\]\(([^)#][^)]*)\)`)
	for _, doc := range []string{"README.md", "CONTRIBUTING.md", "docs/guide.md", "CHANGELOG.md"} {
		src := readFile(t, doc)
		found := 0
		for _, m := range link.FindAllStringSubmatch(src, -1) {
			target := m[1]
			if strings.HasPrefix(target, "http") {
				continue
			}
			found++
			if i := strings.Index(target, "#"); i >= 0 {
				target = target[:i]
			}
			if target == "" {
				continue
			}
			if _, err := os.Stat(target); err != nil {
				t.Errorf("%s links to %s, which does not exist", doc, target)
			}
		}
		_ = found
	}
}

// The package doc is the first thing pkg.go.dev shows. It has to name the entry
// point, or a reader has to guess how to install the thing.
func TestPackageDocNamesTheEntryPoint(t *testing.T) {
	src := readFile(t, "simdblas.go")
	for _, want := range []string{"blas64.Use", "Implementation"} {
		if !strings.Contains(src, want) {
			t.Errorf("the package doc does not mention %s", want)
		}
	}
}

func readFile(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	return string(b)
}
