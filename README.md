# simdblas

**A BLAS implementation for [gonum](https://github.com/gonum/gonum), backed by
[simd.go](https://github.com/sebishogun/simd). No cgo.**

Gonum's own BLAS is pure Go with SSE2-only assembly on amd64 and none at all on
arm64. This replaces the routines where a vector unit helps, reaching AVX-512,
NEON, SVE2, RVV, VSX, VX and LASX — with no cgo, because the kernels are
compiled ahead of time and committed as assembly.

## Install

```
go get github.com/sebishogun/simdblas
```

One call at startup, and everything above BLAS — gonum's `mat`, `stat`,
`optimize` — runs on it:

```go
import (
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"
	"github.com/sebishogun/simdblas"
)

func init() {
	blas64.Use(simdblas.Implementation{})
	blas32.Use(simdblas.Implementation{})
}
```

`blas32` and `blas64` are separate process-wide backends. Register both if the
application uses both precisions, and let the application own registration;
reusable libraries should not change a caller's global BLAS implementation.

Requires Go 1.25 or later and `github.com/sebishogun/simd v1.20.0`.

## Numbers

These are the v1.0.1 release measurements: Ryzen AI MAX+ 395 (Zen 5, AVX-512,
32 threads), float64, both implementations
in the same process on the same inputs. `-count 6`, twice, reporting the worse
of the two runs, at a load average under 1 — see [measuring](#measuring) before
believing any of it.

| routine | size | gonum | simdblas | |
|---|---|---|---|---|
| `Dnrm2` | 1,000 | 804 ns | 28 ns | **29.1×** |
| | 100,000 | 80.2 µs | 3.62 µs | **22.2×** |
| | 1,000,000 | 807 µs | 53.2 µs | **15.1×** |
| `Dgemv` | 2048² | 985 µs | 104 µs | **9.5×** |
| `Dgemm` | 64² | 40.8 µs | 4.37 µs | **9.3×** |
| | 256² | 416 µs | 74.1 µs | **5.6×** |
| | 512² | 2.10 ms | 0.50 ms | **4.2×** |
| | 128² | 109 µs | 33.6 µs | **3.2×** |
| `Daxpy` | 1,000 | 125 ns | 37 ns | 3.4× |
| `Ddot` | 1,000 | 108 ns | 33 ns | 3.2× |
| `Dgemv` | 1024² | 84.5 µs | 33.5 µs | 2.5× |
| | 256² | 4.37 µs | 2.65 µs | 1.6× |
| `Daxpy` | 1,000,000 | 140 µs | 110 µs | 1.3× |
| `Ddot` | 1,000,000 | 131 µs | 108 µs | 1.2× |
| `Dsymm` | 512² | | | **18.4×** |
| `Dsyr2k` | 256² | | | **11.8×** |
| `Dsyr` | 512 | | | 3.3× |

No v1.0.1 release row was slower than gonum. The narrowest routine case was
1.13× (`Ddot` at 100,000 elements, where the bus is the limit rather than the
arithmetic) and the narrowest workload was 1.00×. The committed benchmark
harness covers `Ddot`, `Daxpy`, `Dnrm2`, `Dgemm` and `Dgemv`; the symmetric,
decomposition and whole-workload rows in this README are release-time
measurements, not continuous benchmark gates.

`Dnrm2` is the outlier because gonum computes it with a scaling loop that cannot
vectorize. `Dgemm` and `Dgemv` use simd.go's parallel kernels, which is why they
beat a gonum implementation that is itself parallel.

## What is accelerated

The routines where whole-slice vector work pays, in both precisions:

- **Level 1** — `dot`, `nrm2`, `asum`, `axpy`, `scal`, `swap`, `rot`
- **Level 2** — `gemv`, `ger`, `syr`, `syr2`
- **Level 3** — `gemm`, `symm`, `trsm`, `trmm`, `syrk`, `syr2k`

Everything else is inherited from `gonum.Implementation`. That is the design:
`simdblas.Implementation` embeds gonum's implementation, so every BLAS method
exists before an accelerated override is selected. Delegated calls are gonum's
own implementation, including its invalid-argument panics. Accelerated calls
are tested against gonum within numerical tolerances and may differ in the last
bits.

The direct matrix-kernel path requires **no transpose, alpha of 1, beta of 0 and
natural leading dimensions**. That is not the full accelerated surface. Larger
`gemm` calls outside that shape can pack transposed, scaled or windowed operands
into pooled contiguous scratch. `trsm`, `trmm`, `syrk`, `symm` and `syr2k` use
blocked or densified paths when their work thresholds are met. Level 1 and
`gemv` retain the narrower unit-stride and contiguous guards.

## When it hands the call back to gonum

This is the part that decides whether swapping the backend can make a workload
slower. Three things route a call to gonum's own implementation.

**The call is outside an accelerated path.** Level 1 requires unit strides;
`gemv` also requires a contiguous, untransposed matrix with alpha 1 and beta 0.
General `gemm` accepts transposes, scaling and matrix windows only when there is
enough arithmetic to repay packing. The blocked Level 3 routines have their own
shape and work guards. Unsupported routines and invalid arguments delegate, so
gonum retains their behavior and panic messages.

**The problem is too small.** The kernels have their own threshold — under a
few dozen elements they run a plain Go loop, because crossing into assembly
costs more than it saves. That is the right fallback inside a general-purpose
library and the wrong one here, because gonum's Level 1 is *hand-written SSE2*
rather than a Go loop. Falling back to portable code would hand gonum the win at
small sizes, so instead the call is delegated. Measured, gonum against this,
before the thresholds existed:

| n | asum | axpy | dot | nrm2 | scal |
|---|---|---|---|---|---|
| 8 | 0.52× | 0.62× | 0.46× | 1.09× | 0.55× |
| 16 | 1.04× | 0.72× | 0.57× | 2.15× | 0.59× |
| 32 | 1.75× | 0.97× | 0.77× | 3.70× | 0.98× |
| 64 | 3.62× | 1.29× | 1.15× | 6.47× | 1.07× |
| 256 | 14.03× | 2.42× | 2.26× | 18.75× | 1.79× |

So each routine has a minimum length, set at the first size that is clearly
ahead rather than at break-even: **dot 64, axpy 48, scal 48, swap 48, rot 48,
asum 16, nrm2 8, syr 128**. `gemm` and `trsm` have work thresholds instead,
`gemm` also a ratio of arithmetic to data movement, and `ger` one on row
length — same reason, measured the same way.

**The answer would be wrong.** `nrm2` is the only case: BLAS defines it to avoid
spurious overflow, so when the sum of squares overflows or sinks into the
subnormals the call goes to gonum's scaled algorithm. See below.

**What delegation costs.** One extra method call — measured at **+0.4 to +0.7
ns** on a 3–6 ns operation. That is the floor on how much slower this can be
than gonum for any call, and it only applies below the thresholds.

## Results differ from gonum's, and that is the point

BLAS does not specify bit-exact results and no two implementations agree.
Reductions here use a fixed sixteen-accumulator tree and gonum's use a four-way
unrolled loop, so a dot product of 4096 elements typically differs in the last
unit in the last place.

The difference worth knowing runs the other way:

```
gonum Ddot   0x...de75   with its amd64 assembly
gonum Ddot   0x...de61   without it
simdblas     0x...de71   both
```

**Gonum's answer depends on what it was compiled for.** It has no arm64
assembly, so the same program can give different results on x86 and arm64. The
accelerated reductions here use a fixed accumulation order rather than one that
follows vector width. This repository tests determinism in the selected tier;
cross-tier and cross-architecture kernel agreement is supplied by `simd
v1.20.0`.

## One place the specification bites

`nrm2` is *defined* by BLAS to avoid spurious overflow and underflow. The
obvious implementation — `sqrt` of the sum of squares — does not: a vector of
1e200s returns `+Inf`, and a vector of 1e-200s returns zero, where gonum returns
the correct finite value for both.

So the fast path runs and its own failure is detected. The sum of squares
overflowing gives `+Inf`; sinking into the subnormals gives a value below the
smallest normal float, which is the case that is easy to miss, because a vector
of 1e-160s squares to 1e-320 — not zero, so a test for zero passes it through
with six digits of mantissa already gone. Both go to gonum's scaled algorithm.

The common path pays one comparison. Checking the magnitudes up front instead
was measured at 483 µs against 54 µs for a million elements, which spends most
of the win to buy an answer that is almost always already correct.

## Testing

Every override is checked against the gonum routine it replaces, at two
different bars. Where the fast path does **not** apply this delegates, so the
answer must be gonum's exactly, to the bit — anything else means a guard is
wrong and a fast path ran when it should not have. Where it does apply, the bar
is a relative tolerance.

There is also a test asserting the accelerated path is *reached*: a suite that
passed by delegating everything would look identical to one that worked, so
`TestFastPathActuallyRuns` requires `Ddot` and `Dnrm2` **not** to return gonum's
exact bits. `Dgemm` cannot be checked that way — the blocked kernel and gonum's
both accumulate sequentially over `k` and agree exactly at every size tried — so
its guards are tested directly instead.

```
go test ./...
go test -race ./...
go vet ./...
```

## Measuring

The numbers above are from one machine and they are the only numbers here.
Benchmark noise is one-sided, so the minimum of several runs is what gets used
rather than the mean, and the machine has to be quiet. An earlier draft of that
table was measured while a docker cross-compilation lane was using all 32
threads and reported `Dgemm` at 512 as 2.27× instead of 4.81×.

```
go test -run '^$' -bench . -count 6 .
```

## Decompositions

`mat.LU`, `mat.QR`, `mat.Cholesky` and `mat.Inverse` all run faster. Same
method as the table above — worse of two runs, minimum of four counts:

| | gonum | simdblas | |
|---|---|---|---|
| `QR` n=256 | 2.20 ms | 1.11 ms | **1.98×** |
| `Inverse` n=512 | 19.3 ms | 12.5 ms | **1.54×** |
| `Cholesky` n=256 | 738 µs | 490 µs | **1.51×** |
| `Inverse` n=256 | 2.92 ms | 2.08 ms | **1.40×** |
| `Cholesky` n=512 | 7.97 ms | 5.98 ms | **1.33×** |
| `LU` n=256 | 945 µs | 758 µs | **1.25×** |
| `LU` n=512 | 7.35 ms | 6.00 ms | **1.22×** |
| `QR` n=512 | 14.3 ms | 12.6 ms | **1.14×** |

These were flat at 1.00× until v0.3.0, and the reason is worth stating because
it is not obvious. LAPACK almost never calls BLAS in the shape application code
does: `Dgetrf` multiplies *submatrix windows* with an alpha of −1, `Dgetf2`
passes `ger` a strided column, and half the work is in `trsm` and `trmm`, which
had no fast path at all. Every guard rejected every call.

So the gemm path now packs windowed, transposed and scaled operands into
contiguous scratch and multiplies there — three copies that are O(mk+kn+mn)
against O(mnk) of arithmetic, which is 2.5% of the multiply at LAPACK's own
blocking. `trsm`, `trmm` and `syrk` are blocked on top of it: gonum solves the
small diagonal blocks and the accelerated gemm does the rectangular updates
between them.

`trmm` needed one more case. LAPACK's `Dlarfb` applies a block reflector
through a triangle that is exactly the block size — 64 across — against a B as
tall as the matrix, and blocking a 64-wide triangle produces one block and no
gemm. Below the block size the triangle is filled out into a full matrix and
handed to gemm whole. That is what took QR from 1.05× to 1.98× at n=256.

### What is left on gonum's code, and why

The **banded and packed storage variants**, and **everything complex**. Neither
has been attempted; gonum's `mat` rarely produces the first, and the second is a
larger job than it looks.

**`symv` and `trmv` were written, measured and deleted.** The same densify trick
that makes `symm` 18× loses on these Level 2 calls. Level 3 does n³ of arithmetic
on n² of data, so filling in the implied half of a matrix is free next to the multiply.
Level 2 does n² on n², so the densify is the *same order* as the work it is
preparing — and it replaces gonum reading n²/2 stored elements with n² of
branchy copying plus a full matrix-vector multiply. Measured 0.29× at n=192 and
0.11× at n=1024, so they are not here.

**`trsv` is not attempted.** A triangular solve is sequential — `x[i]` depends
on `x[0..i-1]` — so it is a scan rather than a map, and Level 2 has no work to
amortise a blocked algorithm against.

## Whole workloads

Routine-level numbers do not tell you whether a program gets faster. These are
the end-to-end ratios recorded for the v1.0.1 release; the absolute times from
an earlier run are omitted rather than mixed with the later ratios.

| workload | simdblas / gonum speed |
|---|---:|
| covariance + Cholesky, 5000×300 | **4.36×** |
| covariance + Cholesky, 5000×100 | **2.11×** |
| dense inference, 128×512×1024 | **1.80×** |
| least squares by QR, 5000×200 | **1.52×** |
| least squares by QR, 2000×50 | 1.00× |

The last row is the useful one. It was **0.73×** until v0.5.0 — no individual
routine was slow, and the combination was, because applying a QR factor to a
single right-hand side produces a multiply whose packing costs more than its
arithmetic. A size threshold does not catch that; a ratio of arithmetic to data
movement does.

## Documentation

- **[docs/guide.md](docs/guide.md)** — installing it, what gets faster and what
  does not, how to measure it on your own machine, and what to do when
  something looks wrong.
- **[Runnable examples](https://pkg.go.dev/github.com/sebishogun/simdblas#pkg-examples)**
  — every entry point, compiled and checked by `go test`.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — the bar for a change, and how to add
  an accelerated routine.
- **[CHANGELOG.md](CHANGELOG.md)**.

## Status

The latest tagged and published release is **v1.0.1**. The current main branch
uses `simd v1.20.0`; that dependency update is not yet tagged as a simdblas
release. The API is one exported type and is stable: `Implementation` stays an
embedding of `gonum.Implementation`. Every BLAS interface remains satisfied,
and delegated routines remain gonum's own methods, panics included.

What is *not* promised: which routines are accelerated and where the thresholds
sit — both are measurements and should move when a better one exists — and the
last unit in the last place across versions, since a call moving between the
accelerated and delegated path accumulates differently. Within a version the
result is deterministic and identical on every architecture. See
[CHANGELOG.md](CHANGELOG.md).

Measured on amd64 only. The `simd v1.20.0` dependency has real-hardware
correctness coverage on amd64 and arm64 NEON and emulated correctness coverage
for its other targets; that does not establish simdblas performance there. This
package has only been timed on the Zen 5 machine above, and the thresholds are
measured constants. Reports from other architectures are the contribution it
most needs — see
[CONTRIBUTING.md](CONTRIBUTING.md).


## The rest of the family

The maintained inventory of libraries built on `simd` is in the
[`simd` README](https://github.com/sebishogun/simd#built-on-this). Platform,
performance and release status stay with each repository instead of being
copied here.

## License

MIT — see [LICENSE](LICENSE). Depends on
[simd.go](https://github.com/sebishogun/simd) (MIT) and
[gonum](https://github.com/gonum/gonum) (BSD-3-Clause).
