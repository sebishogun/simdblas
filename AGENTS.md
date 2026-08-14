# AGENTS.md — working on simdblas

Instructions for coding agents. Read the required documents before changing
anything; this file is the summary, the documents are the detail.

## Core tenets: performance-aware programming

**These are the core tenets of this codebase. Read them before writing a line.**

The stance is Casey Muratori's: *performance-aware programming*. Not
"optimization" as a phase that happens after the code works — knowing roughly
what the machine will do with what you write, while you write it. The
alternative is not "clean code that gets optimized later"; it is code whose
shape forecloses the fast version, and the rewrite costs more than thinking for
five minutes did.

Two ideas underneath everything below:

- **Know the order of magnitude before you type.** How many times does this run
  — once, per request, per row, per element? What does one iteration touch?
  Nobody needs a cycle count; everybody needs to know whether they just wrote
  something that runs 200,000 times and allocates.
- **The machine is not an abstract machine.** It has caches, a prefetcher, wide
  registers, and many cores. Code that pretends otherwise leaves 10-100x on the
  floor, and no amount of later profiling recovers a layout decision.

**How the tenets relate.** They are not a list of independent good ideas. The
data-layout ones exist to make the bulk operation POSSIBLE:

    struct-of-arrays + grouped lifetimes + zero per-element allocation
        -> contiguous, uniformly-typed arrays
            -> the kernel can run at all
                -> SIMD, and the parallel shard boundaries come free

You cannot vectorize an array-of-structs: the lanes are not adjacent. You
cannot vectorize a slice that is really a graph of separately-allocated
objects. You cannot keep a kernel fed if every element costs an allocation.
So struct-of-arrays, arenas and lifetime grouping are not housekeeping to do
after the fast path works — they are the precondition for the fast path
existing, and the reason a layout decision made carelessly cannot be recovered
by profiling later.

Read the sections in that order, and design in that order.

### 1. Zero allocations wherever it is possible at all

Not "few" — zero, on any path that runs per element, per record, per row or per
request.

The checklist, in the order it usually pays:

- **Nothing per-element or per-record that can be per-batch.** A `map` built
  per record, a `fmt.Sprintf` per line, a `[]byte`->`string` per field: at 200k
  records those are 200k allocations and 200k pieces of GC work. Reach for a
  byte scan over the fixed shape instead of a reflective decode into a map.
- **Size every slice and map you can size.** `make([]T, 0, n)` when n is known
  or estimable. Growing from nil reallocates and copies the whole thing at
  every doubling.
- **Reuse the caller's buffer.** Append into a supplied `[]byte`, compact in
  place when the write cursor provably trails the read cursor, take a `dst`
  parameter rather than returning a fresh allocation.
- **Do not scan twice.** If a later stage already parses the data, do not
  validate it fully first — do the O(1) structural check and let the one place
  that parses report the rest.
- **Escape analysis is part of the design.** A pointer stored in an interface,
  a closure capturing a local, a returned slice of a local array: each forces a
  heap allocation. `go build -gcflags=-m` says which.
- **Prefer a wider type to a pointer chase.** An index into a slab beats a
  pointer when the slab is contiguous — it is smaller, it does not escape, and
  it keeps the array vectorizable.

Verify with `-benchmem`. `0 allocs/op` is a target you can actually hit on a
hot path, and worth stating in the doc comment when you hit it.

### 2. Think about the data, then the code

Muratori's central point, and the one most often skipped. The layout of the
data decides the speed; the instructions are usually a detail.

**Struct-of-arrays over array-of-structs** for anything scanned columnwise. A
filter that reads one field should stream that field's array, not stride
through whole records pulling in fifteen fields it does not want. This is the
single highest-leverage decision in a columnar store, and it is made when the
type is declared, not later.

**Group lifetimes; allocate them together.** Objects born together and dying
together belong in one allocation. A per-request arena — one buffer that
everything for that request is carved out of, released in one move when the
request ends — replaces thousands of individual allocations and frees with a
single pointer reset. It also gives locality for free: everything the request
touches is contiguous. Where the lifetime is per-batch, per-group or
per-connection instead, the same applies at that scope. The rule is that the
allocation boundary should match the lifetime boundary; when it does not, you
get either leaks or a per-object free.

**Use the whole cache line.** Touch it once and consume all of it. Block a pass
to fit L1/L2 rather than striding across a large array repeatedly. Keep hot
fields adjacent and cold fields elsewhere so a line carries only what the loop
reads. Watch for false sharing when threads write adjacent words.

Locality is a hypothesis to check with perf counters, not a rule to apply
blindly: windowing won in simdcsv and did nothing in simdjson.

### 3. Do the work in bulk — use SIMD

This family exists for it. Whole-slice work goes through the kernels, not a
hand-written scalar loop. Where no kernel exists for the shape, say so
explicitly rather than quietly writing the scalar loop and leaving it.

Check the dispatch actually reaches the kernel at runtime: every complex kernel
in `simd` was dead code from v1.14.0 to v1.20.0 because nothing walked the
tables the runtime indexes.

A per-element function call defeats vectorization outright — measured at 11
extra instructions per element, a 2.56x ratio. If the API shape forces one, the
API shape is the bug.

### 4. Don't do the work at all

The fastest code is the code that does not run. Prune before you decode: a
bloom filter that rejects a group, a time window that skips a block, a column
never materialized because nothing asked for it. simdlogs' rare-needle path
beats a full scan by rejecting groups without decoding them, not by decoding
faster.

Hoist invariants out of loops. Compute once what does not change. Do not scan
twice — if a later stage already parses the data, do the O(1) structural check
and let the one place that parses report the rest.

### 5. Multi-threaded where it is beneficial

And only there. Parallelism pays when the work per shard clears the
coordination cost; below that it is slower and less predictable.

Shard on a boundary the data already has (groups, blocks, row ranges), give
each worker its own output buffer, merge once. Never share a mutable buffer
between goroutines without saying so in the doc comment. `-race` is a gate.

### 6. `sync.Pool` is the last resort — and it has to be correct

Reach for it last. Most allocation wins are a size hint, an arena, or a
caller-supplied buffer: free at runtime, no correctness hazard. A pool costs
Get/Put, a miss allocates anyway, and it introduces a class of bug the others
cannot have.

When a pool IS the right answer, these are not optional:

- **The buffer must be fully overwritten before anything reads it.** A pooled
  buffer arrives holding a PREVIOUS request's data. If any path reads an
  element it did not write, that request's data is silently served to this one
  — a correctness bug, cross-request data leakage, not a performance one. Know
  the property holds and say WHY in the doc comment; do not assume it.
- **Prove it with a poisoning test.** Fill pooled buffers with a value that
  cannot occur, then assert the pooled result equals the unpooled result
  exactly. Write that test FIRST. This is the only thing that catches the bug,
  because the unpooled path zeroes and therefore hides it.
- **Ownership must be unambiguous.** A pooled buffer must not escape into a
  returned value, be captured by a goroutine that outlives the Put, or be
  aliased by a slice the caller keeps. Returning a slice of a pooled array is a
  use-after-free in all but name.
- **Put back exactly what you took**, reset to a known state, once. A double
  Put hands the same array to two callers at the same time.
- **Pool a pointer, not a slice.** A `[]T` placed in an `any` allocates on
  every Put, which is the cost the pool exists to remove.
- **Sizing is part of the contract.** A pool of mixed sizes either wastes the
  large buffers or reallocates on the small ones; decide which and say so.
- **Testing note:** `sync.Pool.Put` drops the value at random one time in four
  under `-race`, so any test asserting reuse across a single round trip is red
  a quarter of the time. Assert reuse within a few attempts, not on a
  particular one.

### Then measure

These tenets are where to start, not a substitute for the benchmark.
Fast-looking code that was never measured is a guess. The noise floor, the
interleaved A/B discipline, and "disassemble before you theorise" apply to code
written this way exactly as they apply to a tuning change — and a claim with no
number behind it does not go in a doc.

## Product and boundary

simdblas is a BLAS implementation for gonum, backed by
github.com/sebishogun/simd. **No cgo** — the kernels are compiled ahead of
time and committed as assembly.

- The whole exported surface is one type: `Implementation struct{
  gonum.Implementation }`. The four BLAS interfaces are satisfied by
  construction (tested by `TestSatisfiesBLAS`).
- Callers install it at startup with `blas64.Use(simdblas.Implementation{})`
  and `blas32.Use(...)`. Registration is **caller-owned and process-global**:
  a library must never change a caller's BLAS, and both precisions are
  separate backends.
- **Delegation defines compatibility.** Every override guards its fast path
  and hands everything else — wrong shape, too small, invalid arguments — to
  the embedded gonum method. Delegated calls are gonum's own code, panics
  included. Invalid arguments must never be rejected here.
- Non-goals: accelerating everything; bit-exact agreement with gonum on
  accelerated paths; last-bit reproducibility across versions; performance
  claims off amd64; changing a caller's global BLAS from a library.

## Current status

- Latest published release: **v1.0.1**. `main` carries the `simd v1.20.0`
  dependency bump, not yet tagged as a simdblas release.
- Which routines are accelerated and where the thresholds sit are
  **measurements, not promises**, and may move in a minor release. Within a
  version results are deterministic and identical on every architecture.
- The roadmap is **not shipped**. Nothing in docs/roadmap.md or
  docs/plans/ exists in the code until it has passed the production bar
  (below). Do not document planned routines as if they shipped.

## Required reading, in order

0. **The core tenets above** — performance-aware programming. Every
   agent working in this repository reads them before writing code.
1. `README.md` — the shipped front page, numbers, status.
2. `docs/guide.md` — usage, what gets faster, measuring.
3. `docs/architecture.md` — package boundary, data flow, promises.
4. `docs/lld/delegation-and-guards.md` — every accelerated routine, guard and
   threshold.
5. `docs/lld/level-2-and-level-3.md` — how the matrix paths are built.
6. `docs/verification.md` — the gates and what counts as a number.
7. `docs/roadmap.md` — evaluated future work, not promises.
8. `docs/wrong.md` — measurements that argued against changes.
9. `CHANGELOG.md` — release history and the v1.0 promise.
10. `CONTRIBUTING.md` — the bar for a change.
11. `docs/plans/*` — before doing any planned work.

## Rules

- **Disassemble first.** Before proposing a cause for anything slow, before
  writing a variant, before reading a profile delta:

  ```
  go test -c -o /tmp/x.test .
  go tool objdump -s 'simdblas\.FunctionName' /tmp/x.test
  ```

  Register spills, missed bounds-check elimination, index multiplies,
  inlining and `memmove` calls only show up there.
- **The 8.3% floor.** Code-layout noise here is 8.3%; anything smaller cannot
  be told from nothing by wall clock, and more samples do not help. Below the
  floor, judge with `perf stat -e instructions:u,cycles:u` (layout-independent)
  and the disassembly.
- **Quiet machine, interleaved A/B, minima.** Load average under 1.
  `-count 6`, minimum of runs, comparison run twice, worse of two runs
  reported, both implementations in one process on the same inputs — never
  across sessions.
- **Bare gates.** Never pipe a gate through `tail` (or anything else) without
  `set -o pipefail` — the pipe launders failures into green exits. Run gates
  bare: `go test ./... && go test -race ./... && go vet ./... && git diff --check`.
- **Numerical.** Accelerated answers may differ from gonum's in the last bits
  (fixed accumulation order); delegated answers must be gonum's bits exactly.
  `nrm2` is the one numerical fallback (overflow/subnormal → gonum's scaled
  algorithm). The untargeted triangle of symmetric/rank-k routines must come
  out exactly as it went in.
- **Concurrency.** gemm and gemv use simd's parallel kernels; scratch comes
  from `sync.Pool` and never escapes the call. The race suite is a CI gate.
- **New accelerated routine:** guard + delegate (invalid arguments included),
  differential tests at both bars, a `TestFastPathActuallyRuns`-style
  engagement test (or direct guard tests where results cannot distinguish —
  see `TestGuards`), minimum-work/length set at the first clearly-ahead size,
  disassembly, then measurement.
- **Production bar:** gonum differential correctness, fast-path engagement,
  no regression for guarded shapes, disassembly, performance outside the 8.3%
  floor.
- **Release gate.** A release ships only from a state where the full current
  gate set is green (`go test ./...`, `go test -race ./...`, `go vet ./...`),
  the differential/fast-path/conformance and architecture/benchmark evidence
  required by docs/verification.md is recorded, and the status, tag, CHANGELOG
  and docs agree with the diff being released. Roadmap work is not shipped:
  nothing from docs/roadmap.md or docs/plans/ lands in a release until it has
  passed the production bar above.
- **wrong.md policy.** Measurements that argue against a change belong in
  `docs/wrong.md`, append-only, whether or not any code changed; never invent
  a measurement; never edit or delete an old entry — add a new one.
- **Docs-only changes** touch only `.md` files: never Go source, tests,
  `go.mod`/`go.sum`, workflows, or benchmark artifacts in the same change.
- **Verification before claiming done.** Run the gates, confirm output, then
  state the outcome. Never claim green without running it.
