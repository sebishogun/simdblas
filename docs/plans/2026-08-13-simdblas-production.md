# simdblas Production Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute the source-backed evaluations in docs/roadmap.md — each either shipping with tests or being deleted with its measurement recorded in docs/wrong.md — and release the untagged `simd v1.20.0` dependency state on main.

**Architecture:** Every code task is an evaluation on a branch with a delete-or-record outcome: write the differential/engagement test for the candidate path first (it fails, because the guard delegates and the path does not exist), prototype against the guard, measure under the project's rules, then either keep with tests and docs or delete and record in docs/wrong.md. Nothing is scheduled as certain to ship; the only unconditional task is the release of the current untagged state.

**Tech Stack:** Go 1.25, github.com/sebishogun/simd v1.20.0 (the vector kernels), gonum.org/v1/gonum v0.17.0 (blas interfaces + `blas32`/`blas64` registration), gonum's differential oracle in diff_test.go, `go test`/`go vet`, `go tool objdump`, `perf stat -e instructions:u,cycles:u`, GitHub Actions (ci.yml).

---

## Ground rules for every task

- **Production bar (from docs/roadmap.md):** a candidate ships only after (1) gonum differential correctness at both bars, (2) fast-path engagement, (3) no regression for guarded shapes, (4) disassembly, (5) performance outside the 8.3% layout floor on a quiet machine with interleaved A/B minima.
- **Disassembly before any performance hypothesis:**

  ```
  go test -c -o /tmp/x.test .
  go tool objdump -s 'simdblas\.Dsyrk' /tmp/x.test
  ```

- **Bench rule:** `go test -run '^$' -bench . -count 6 .` on a quiet machine (load average under 1), minima compared, interleaved in one session; below the 8.3% floor judge on `perf stat -e instructions:u,cycles:u` and the disassembly.
- **Gates are run bare** — never `go test ./... | tail` without `set -o pipefail`; the pipe launders failures into green exits.
- **A losing evaluation commits two things:** the deleted prototype and a docs/wrong.md entry with the measurement and its provenance.
- Commit after each task; commit messages match the repo style (lowercase `feat:`/`docs:`/`test:` prefixes).

---

### Task 1: Release the current untagged state

**Files:**
- Modify: `CHANGELOG.md` (new version entry)
- Modify: `README.md` only if the Status section's version string changes

**Step 1: Establish the diff**

Run: `git diff v1.0.1..HEAD --stat`
Expected: the range contains the `simd v1.2.0 -> v1.20.0` dependency bump plus
documentation commits — including the README family-table link update, the
guide refresh and the production-architecture docs — and nothing else; the
commit count grows as review commits accumulate and is not fixed. The
important conclusion to retain: no routine implementation changed, so the
diff is dependency and documentation only.

**Step 2: Choose the version**

Rule: dependency bump with no routine change → patch release (`v1.0.2`). Any routine change → minor (`v1.1.0`). State the choice and the reason in the commit message.

**Step 3: Write the CHANGELOG entry**

Add the entry above the v1.0.1 block: what the dependency bump brings (kernels and correctness coverage — quote go.mod), that the API and accelerated surface are unchanged.

**Step 4: Run the release gates**

Run: `go test ./... && go test -race ./... && go vet ./... && git diff --check`
Expected: all green, no output from `git diff --check`.

**Step 5: Commit and tag**

```
git add CHANGELOG.md [README.md]
git commit -m "docs: changelog for v1.0.2 (simd v1.20.0)"
git tag v1.0.2
git push origin v1.0.2
```

Expected: tag exists; `git describe` shows it.

---

### Task 2: Evaluate a blocked `syrk` that only touches the triangle

**Source:** trmm.go:188-196 — the shipped syrk computes the whole product and keeps half; the code says a blocked form "would be better and is not written".

**Files:**
- Create: `syrk.go` (the prototype; if it ships, move `Dsyrk`/`Ssyrk` bodies out of trmm.go/level3_f32.go — or keep them if the prototype loses and they are deleted)
- Create: `syrk_test.go` (differential + triangle-invariant tests)
- Modify: `bench_test.go` (add `BenchmarkDsyrk`, mirroring `BenchmarkDgemm`: sizes 64/128/256/512, `k == n`, simd and gonum sub-benchmarks in one process)

**Step 1: Write the failing test**

In `syrk_test.go`, write `TestSyrkBlockedMatchesGonum` calling a not-yet-existing `impl.syrkBlocked(ul, t, n, k, alpha, a, lda, beta, c, ldc)` for both uplos, both transposes, `n, k` in {64, 129, 200}, comparing against `refImpl.Dsyrk` with the 1e-10 bar, plus the other-triangle-untouched check with `-777` sentinels (pattern: symmetric_test.go).

**Step 2: Run it to verify it fails**

Run: `go test -run TestSyrkBlockedMatchesGonum .`
Expected: FAIL — `impl.syrkBlocked undefined`.

**Step 3: Prototype**

Block over `n` in `trsmBlock` (64) chunks: partition the outer product so each output block `C_ij` is a gemm over the k dimension, diagonal blocks and triangle-only writes, folding into `c` with `beta` respected. Reuse `gemmGeneral` and the `sync.Pool` scratch — do not add allocations.

**Step 4: Verify the test passes and the fast path engages**

Run: `go test -run 'TestSyrkBlockedMatchesGonum|TestSymmetricRoutinesLeaveOtherTriangleAlone' .`
Expected: PASS. Add a guard test asserting the blocked path is chosen above the work guard (pattern: TestGuards) and run `go test ./...`.

**Step 5: Disassemble**

Run: `go test -c -o /tmp/x.test . && go tool objdump -s 'simdblas\.(Dsyrk|syrkBlocked)' /tmp/x.test`
Expected: the block loop and gemm calls visible; note any register spills or missed bounds checks before benchmarking.

**Step 6: Measure**

Run: `go test -run '^$' -bench BenchmarkDsyrk -count 6 .` (quiet machine), comparing the prototype against both gonum and the shipped full-product path, interleaved.

**Step 7: Decide**

- Wins outside 8.3% floor at every guarded size, race suite green: keep. Tests + docs (LLD table row for syrk), commit `feat: blocked syrk`.
- Otherwise: delete `syrk.go` and the test, add the measurement to `docs/wrong.md`, commit `docs: record blocked syrk measurement`.

---

### Task 3: Evaluate widening the direct `gemm`/`gemv` fast paths

**Source:** level23.go:18-33 — the direct gemm path is narrow by design and the comment says widening (transposed B, non-zero beta) is "worth doing against measurements rather than in advance".

**Files:**
- Modify: `level23.go` (candidate `gemmFast2`/`gemvFast2` predicates)
- Modify: `fastpath_test.go` (`TestGuards` additions)
- Modify: `diff_test.go` (fast-path bars for the new shapes)
- Modify: `bench_test.go` (transposed/scaled sub-benchmarks)

**Step 1: Measure the gap first**

Add sub-benchmarks for `Dgemm(Trans, NoTrans, …)`, `Dgemm(NoTrans, NoTrans, alpha=0.5, beta=1, …)` and `Dgemv(Trans, …)` at 256²/512², comparing the shipped paths (packed general for gemm, delegated for gemv) against gonum.
Run: `go test -run '^$' -bench 'BenchmarkDgemm|BenchmarkDgemv' -count 6 .`
Expected: a baseline; record it in the task notes before touching guards.

**Step 2: Write the failing guard tests**

Add cases to `TestGuards` asserting the candidate predicate accelerates the new shapes (currently `gemmFast` returns false for them — the new tests fail on the not-yet-existing predicate).

**Step 3: Prototype**

Add `gemmFast2`/`gemvFast2` in level23.go; for gemv, a transposed or scaled path needs scratch (one row-buffer for the transpose or a scale pass) — allocate from the existing pools only.

**Step 4: Verify**

Run: `go test ./...` — differential bars must pass at both precisions; `go test -race ./...` green.

**Step 5: Disassemble and measure**

Repeat the Task 2 steps 5-6 against the baseline from step 1.

**Step 6: Decide**

Keep (with tests, docs, commit `feat: widen gemm direct path` / `feat: widen gemv path`) or delete and record in `docs/wrong.md` with the measurements, commit `docs: record gemm/gemv widening measurement`.

---

### Task 4: Complex feasibility evaluation

**Source:** README — "everything complex" is unaccelerated and "a larger job than it looks".

**Files:** none certain; possibly `level1_complex.go` if kernels exist.

**Step 1: Establish whether simd v1.20.0 has complex kernels**

Run: `go doc github.com/sebishogun/simd | grep -i complex` and grep the simd module in `$GOMODCACHE`.
Expected: either a list of complex kernels or an empty result.

**Step 2a: No kernels — stop and record**

Add a `docs/wrong.md` entry (feasibility finding: complex support requires upstream kernels; not a simdblas code task), update the roadmap E3 line, commit `docs: record complex feasibility finding`. Task ends here.

**Step 2b: Kernels exist — prototype the complex Level 1 reductions**

Write differential tests for `Zdot`/`Zaxpy` (blas.Complex128, two bars, NaN-aware) first; run to verify they fail; implement via the kernels with the Level 1 guard shape (`unit2`); verify; disassemble; measure at 1000/100000/1000000; keep or delete-and-record per the Task 2 decide step.

---

### Task 5: Evaluate `trsv`

**Source:** README — sequential by nature; the roadmap lists it as expected to lose.

**Files:**
- Modify: `level23.go` (candidate `Dtrsv` override prototype)
- Modify: `diff_test.go` (differential test for the candidate)
- Modify: `docs/wrong.md` (outcome, whichever way)

**Step 1: Write the failing differential test**

`TestDtrsvCandidateMatchesGonum` over unit and non-unit `incX`, upper/lower, sizes 8..1024, calling a not-yet-existing `impl.trsvFast(...)`.
Run: `go test -run TestDtrsvCandidateMatchesGonum .`
Expected: FAIL — undefined.

**Step 2: Prototype the best available candidate**

`x[i]` depends on `x[0..i-1]`, so the solve is a scan; the only vectorisable piece is the inner update. Prototype the obvious candidate (per-row axpy through `simd.AddScaled`) and measure it against gonum.

**Step 3: Measure**

`go test -run '^$' -bench . -count 6 .` for a trsv benchmark at 256/1024/2048, interleaved.

**Step 4: Decide**

Expected: loses. Delete the prototype, record the measurement in `docs/wrong.md` (confirming the README claim), close roadmap E5, commit `docs: record trsv measurement`. If it wins outside the floor: keep with tests and the production bar, commit `feat: accelerate trsv`.

---

### Task 6: Evaluate blocked transpose packing

**Source:** gemm.go:72-75 — packing a transposed operand walks down a column (cache-hostile); the code says blocking "has not been measured as mattering against the multiply that follows — worth revisiting if a profile ever says otherwise".

**Files:**
- Modify: `gemm.go` (`packMatrix`)
- Modify: `bench_test.go` or a throwaway profile harness

**Step 1: Profile first**

Profile a transposed-heavy workload (a decomposition that reads `Aᵀ`, e.g. `mat.QR` on 512²) with `go test -bench` + `perf stat` and, if needed, pprof.
Expected: either packMatrix shows up materially or it does not.

**Step 2a: Profile shows nothing — record**

Add the finding to `docs/wrong.md`, commit `docs: record transpose pack profile`.

**Step 2b: Profile supports it — prototype**

Block the transpose copy (destination-row blocks read contiguous source columns), write a differential test for `packMatrix` first (fails on the new signature), implement, verify `go test ./...` and `go test -race ./...`, disassemble, measure the transposed-heavy workload; keep or delete-and-record per Task 2's decide step.

---

### Task 7: Cross-architecture measurement programme (arm64 NEON first)

**Source:** README Status + CONTRIBUTING — amd64-only measurements; arm64 NEON has real-hardware correctness coverage in simd v1.20.0; "reports from other architectures are the contribution it most needs".

**Files:**
- Create: `docs/measurements/2026-<date>-arm64-neon.md` (report)
- Modify: `README.md` (only if a number or threshold moves)
- Modify: `level1.go`/guard constants (only if a threshold moves; each change carries its measurement in the comment, per the style rule in CONTRIBUTING)

**Step 1: Run the suite on arm64 NEON**

On an arm64 machine (or a runner that reports NEON via `go run github.com/sebishogun/simd/cmd/simdinfo`):

```
go test ./...
go test -race ./...
go run github.com/sebishogun/simd/cmd/simdinfo
go test -run '^$' -bench . -count 6 .
```

Expected: differential and race suites green (correctness is already covered by simd's real-hardware arm64 NEON coverage); a benchmark table.

**Step 2: Compare against the README table**

Run: `go test -run TestReadmeThresholdsMatchConstants .` — confirms prose still matches constants, which do not move yet.
Expected: table rows where the arm64 result is outside noise of the amd64 claim get flagged for review.

**Step 3: Decide per threshold**

For each threshold where the crossover measurement on arm64 differs from the amd64 constant by more than the 8.3% floor: move the constant, update its comment with the arm64 measurement, update the README if the number is stated there.
Expected: `go test ./...` still green — the docs test re-verifies prose against the moved constants.

**Step 4: Write the report and commit**

`docs/measurements/2026-<date>-arm64-neon.md`: machine, simdinfo output, the bench table, and per-threshold decisions. Commit `docs: arm64 NEON measurement report`.

---

## Task summary

| # | task | outcome if it loses |
|---|---|---|
| 1 | release untagged simd v1.20.0 state | n/a — unconditional |
| 2 | blocked syrk | delete prototype, record in wrong.md |
| 3 | widen gemm/gemv direct paths | delete prototype, record in wrong.md |
| 4 | complex feasibility | record feasibility finding in wrong.md, stop |
| 5 | trsv | record measurement in wrong.md, close roadmap item |
| 6 | blocked transpose packing | record profile finding in wrong.md |
| 7 | arm64 NEON measurements | report lands; thresholds move only with measurements |

After every task: `go test ./... && go test -race ./... && go vet ./... && git diff --check` bare, then commit. When all tasks are done and the branch is green, finish per superpowers:finishing-a-development-branch.
