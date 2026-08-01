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

Go 1.25 or later.

## Numbers

Ryzen AI MAX+ 395 (Zen 5, AVX-512, 32 threads), float64, both implementations
in the same process on the same inputs. `-count 6`, twice, reporting the worse
of the two runs — see [measuring](#measuring) before believing any of it.

| routine | size | gonum | simdblas | |
|---|---|---|---|---|
| `Dnrm2` | 1,000 | 805 ns | 28 ns | **29.3×** |
| | 100,000 | 80.2 µs | 3.6 µs | **22.3×** |
| | 1,000,000 | 806 µs | 53 µs | **15.1×** |
| `Dgemm` | 64² | 40.7 µs | 4.4 µs | **9.3×** |
| | 256² | 415 µs | 73 µs | **5.7×** |
| | 512² | 2.10 ms | 0.44 ms | **4.8×** |
| `Dgemv` | 2048² | 989 µs | 103 µs | **9.6×** |
| | 1024² | 97 µs | 42 µs | **2.3×** |
| `Ddot` | 1,000 | 107 ns | 33 ns | 3.2× |
| | 1,000,000 | 127 µs | 107 µs | 1.2× |
| `Daxpy` | 1,000 | 125 ns | 37 ns | 3.4× |
| | 1,000,000 | 145 µs | 111 µs | 1.3× |

Nothing measured is slower than gonum; the narrowest case is 1.18×.

`Dnrm2` is the outlier because gonum computes it with a scaling loop that cannot
vectorize. `Dgemm` and `Dgemv` use simd.go's parallel kernels, which is why they
beat a gonum implementation that is itself parallel.

## What is accelerated

The routines where whole-slice vector work pays, in both precisions:

- **Level 1** — `dot`, `nrm2`, `asum`, `axpy`, `scal`, `swap`, `rot`
- **Level 2** — `gemv`, `ger`
- **Level 3** — `gemm`

Everything else is inherited from `gonum.Implementation` and behaves exactly as
it did. That is the whole design: `simdblas.Implementation` *embeds* gonum's,
so every one of the 60-odd BLAS routines exists and is correct from the first
line, and each accelerated one is an override of a working method rather than a
new one that has to be got right.

Acceleration additionally requires **unit strides**, and for the matrix routines
**no transpose, alpha of 1, beta of 0 and natural leading dimensions**. Anything
else — including invalid arguments — falls through to gonum, so the answer is
always a correct BLAS answer and never a wrong fast one, and the panics your
code relies on come from gonum and read exactly as they always have.

That fast-path condition is narrow on purpose for a first release. It is the
shape `mat.Mul` produces for dense matrices that are not views, which is the
common call. Widening it is worth doing against measurements rather than in
advance.

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
assembly, so the same program gives different results on x86 and on arm64. This
implementation gives the same bits on every architecture and every instruction
set, because the accumulation order is fixed rather than following the vector
width. If you need a result that reproduces across machines, that is the reason
to use this.

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

## What this does not accelerate: decompositions

`mat.Mul` is 4.25× faster. `mat.LU`, `mat.Solve`, `mat.Inverse` and `mat.Det`
are **unchanged**, and that was measured rather than assumed.

LAPACK works on submatrix windows with a scaling factor throughout. Its blocked
`Dgetrf` calls `Dgemm` with an alpha of −1 and a leading dimension that is the
parent matrix's width; `Dgetf2` calls `Dger` with a strided column as x; and
`Dtrsm`, the other half of the blocked step, has no fast path here at all. Every
guard below rejects those shapes, so a factorisation runs entirely on gonum's
own code.

That is a real limit rather than a subtlety: the fast paths are shaped for
application code, where matrices are dense and unscaled, and a decomposition is
almost the opposite. Fixing it means accepting a leading dimension and a general
alpha and beta in the `gemm` path, and writing `trsm`. Both are tractable and
neither is done.

The rank-1 update, rotation and swap kernels added in simd.go v1.2.0 are fast
on their own — 8.2×, 13.3× and 4.4× over portable Go — and reachable through
`Dger`, `Drot` and `Dswap` when the arguments are contiguous.

## Status

Early. The interface is satisfied in full and the accelerated subset is tested
against gonum, but this has run on amd64 only. Reports from other architectures
are welcome — simd.go itself is verified on amd64 and arm64 NEON, and under
emulation everywhere else.

## License

MIT — see [LICENSE](LICENSE). Depends on
[simd.go](https://github.com/sebishogun/simd) (MIT) and
[gonum](https://github.com/gonum/gonum) (BSD-3-Clause).
