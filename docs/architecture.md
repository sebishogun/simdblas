# Architecture

How simdblas is put together, and what each layer is allowed to do. This is the
current shipped architecture, not a plan: the exact routines and guards are in
the [LLDs](lld/delegation-and-guards.md), and anything not shipped yet is in the
[roadmap](roadmap.md).

## The package boundary

The whole exported surface is one type (simdblas.go:62-74):

```go
type Implementation struct{ gonum.Implementation }
```

Everything else is unexported. `Implementation` embeds `gonum.Implementation`,
so every BLAS method exists and is correct before any accelerated override is
written, and the four BLAS interfaces are satisfied by construction — asserted
by `TestSatisfiesBLAS` (iface_test.go), which fails the build if that ever
stops being true.

There is no init-time magic: the package does not install itself, does not read
environment variables, and registers nothing. Installation is the caller's one
call at startup:

```go
blas64.Use(simdblas.Implementation{})
blas32.Use(simdblas.Implementation{})
```

`blas32` and `blas64` are separate process-global backends, so both calls are
needed for both precisions. Registration is owned by the application: a
reusable library must not change a caller's global BLAS implementation, because
which BLAS a program uses is the program's decision (README, guide).

## Data flow

Every accelerated method has the same shape (level1.go:13-17):

1. **Guard.** Check the arguments against the shape, stride, scaling and
   minimum-work conditions the fast path needs. The guard is a pure predicate
   (for example `gemmFast`, `gemvFast`, `unit`, `unit2`, `syrOK`, `symmOK`) so
   it can be tested directly.
2. **Accelerated path.** Call a simd.go kernel — directly for the contiguous
   shapes, or through packing, blocking or densifying for the general ones.
3. **Otherwise, delegate.** Call the embedded gonum method with the original
   arguments. The call then behaves exactly as it would have without simdblas,
   panics included.

Delegation is therefore the compatibility boundary: a call this package does
not accelerate *is* a gonum call, so there is no second implementation of the
60-odd routines nobody has measured a win for, and no second set of panic
messages to keep in sync.

## Dependencies

- `github.com/sebishogun/simd v1.20.0` — the vector kernels. They are compiled
  ahead of time and committed as assembly, which is why there is **no cgo**
  anywhere in this package (package doc, README).
- `gonum.org/v1/gonum v0.17.0` — the embedded implementation and the `blas`
  interfaces.
- `golang.org/x/sys v0.47.0` — indirect.
- Go 1.25 or later (go.mod).

## The compatibility and numerical boundary

- **Delegated calls return gonum's bits exactly.** The differential tests
  require bit equality wherever the fast path does not apply; anything else
  means a guard is wrong and a fast path ran when it should not have
  (diff_test.go, CONTRIBUTING).
- **Accelerated calls return a different, equally valid BLAS answer.** Reductions
  here use a fixed sixteen-accumulator tree and gonum's use a four-way unrolled
  loop, so the last unit in the last place normally differs. The bar is a
  relative tolerance (1e-12 in the main differential suite).
- **Within a version, the result is deterministic and identical on every
  architecture**, because the accumulation order is fixed rather than following
  the vector width. Cross-tier and cross-architecture agreement is supplied by
  simd v1.20.0 (README, fastpath_test.go).
- **Across versions, nothing is promised about the last bits**: a call may move
  between the accelerated and the delegated path, which accumulate differently.
  Pin a version if you diff floating-point output between builds (CHANGELOG
  v1.0.0).
- **Which routines are accelerated and where the thresholds sit are
  measurements, not promises**, and both may move in a minor release. The
  current values are recorded in the LLDs. Only the Level 1 minima and `syr`
  stated in the README are machine-checked against the constants
  (`TestReadmeThresholdsMatchConstants` parses README.md, not the LLDs); the
  LLD values for `ger`, gemm work/intensity, the trsm block size and the
  syrk/syr2k guards are checked against the source by hand, not by a test.
- **No performance claims off amd64.** Everything in the README was measured on
  one Zen 5 machine; the thresholds are measured constants (README Status).

## What is not here

- No cgo.
- No last-bit or threshold stability promises.
- No performance claims for any architecture that has not been timed.
- No accelerated `symv`, `trmv` (measured and deleted — see
  [wrong.md](wrong.md)) or `trsv`; no banded, packed or complex routines.
  These run gonum's code unchanged, and that is a decision with a measurement
  behind it, not an oversight (README, CHANGELOG v1.0.0).
