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
- **Level 3** — `gemm`, `trsm`, `trmm`, `syrk`

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

Still on gonum's code: `symm`, `syr2k`, the banded and packed storage variants,
the triangular Level 2 routines, and everything complex.

## Status

Early. The interface is satisfied in full and the accelerated subset is tested
against gonum, but this has run on amd64 only. Reports from other architectures
are welcome — simd.go itself is verified on amd64 and arm64 NEON, and under
emulation everywhere else.

## License

MIT — see [LICENSE](LICENSE). Depends on
[simd.go](https://github.com/sebishogun/simd) (MIT) and
[gonum](https://github.com/gonum/gonum) (BSD-3-Clause).
