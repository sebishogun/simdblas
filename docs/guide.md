# Guide

How to use simdblas, what it will and will not speed up, and how to tell the
difference on your own machine.

## Installing it

There is one integration point. `blas64.Use` swaps the implementation every
gonum package reaches for, so a single call at startup covers `mat`, `stat`,
`optimize` and anything else built on them.

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

`blas32` and `blas64` are separate globals. Registering one does nothing for the
other, and float32 code is easy to forget.

Put this in `main`, or in an `init` in your main package. Do not put it in a
library: which BLAS a program uses is the program's decision, and a dependency
that changes it out from under the application is the kind of surprise that
takes a day to find.

## What gets faster

The short answer is anything with enough arithmetic per byte to keep a vector
unit busy, which in practice means the matrix routines.

```go
var c mat.Dense
c.Mul(a, b)          // 4.2x at 512x512
```

Matrix multiplication is the best case because it does O(n³) work on O(n²) data,
so each element is read many times and the memory system is not the limit.
Vector-vector operations touch each element once and are bound by the bus
instead — `Ddot` over a million elements is 1.2×, not 4×, and no implementation
will do much better.

Factorisations sit in between:

| | |
|---|---|
| `mat.QR` | 1.98× at n=256 |
| `mat.Inverse` | 1.54× at n=512 |
| `mat.Cholesky` | 1.51× at n=256 |
| `mat.LU` | 1.25× at n=256 |

They gain less than a bare multiply because a decomposition also contains
scalar work — pivot searches, column norms, the diagonal blocks — that no
vector unit reaches.

## What does not get faster

The banded and packed storage variants, triangular Level 2 routines, `symv`,
`trmv`, and everything complex run gonum's code unchanged. `symm` is accelerated
at sizes where densifying its stored triangle pays for the multiply that
follows. `syr2k`, like `syrk`, computes a full product and folds the requested
output triangle back.

Small problems also do not, and that is deliberate rather than an oversight.
Below a measured threshold each routine hands the call to gonum, whose Level 1
is hand-written SSE2 and beats a general-purpose kernel at a few dozen elements.
The thresholds are in the README.

## Telling whether it helped

Measure the thing you actually run, not a microbenchmark:

```go
func BenchmarkMine(b *testing.B) {
	for _, impl := range []struct {
		name string
		use  func()
	}{
		{"gonum", func() { blas64.Use(gonum.Implementation{}) }},
		{"simdblas", func() { blas64.Use(simdblas.Implementation{}) }},
	} {
		b.Run(impl.name, func(b *testing.B) {
			impl.use()
			for b.Loop() {
				// the work you care about
			}
		})
	}
}
```

Both implementations in one process, on the same inputs. Comparing two builds
compares two builds.

Then: `-count 6` at least, take the **minimum** rather than the mean, and run
the whole comparison twice. Benchmark noise is one-sided — interference only
ever makes a run slower — so the fastest run is the one least interfered with,
and a mean is mostly a measure of how busy the machine was. An earlier draft of
this project's own numbers was measured while a container was using every core
and reported a 4.81× as 2.27×.

## Numerical differences

Results will not match gonum bit for bit. BLAS does not require them to, and no
two implementations agree: reductions here use a fixed sixteen-accumulator tree
and gonum's use a four-way unrolled loop, so a dot product of a few thousand
elements typically differs in the last unit in the last place.

If that matters to you, the difference worth knowing runs the other way. Gonum's
answer depends on what it was compiled for — its amd64 assembly and its pure-Go
fallback disagree, and it has no arm64 assembly at all, so the same program
gives different results on x86 and on arm64. This implementation fixes the
accumulation order rather than following the vector width, so a result computed
here reproduces on any architecture.

For a test suite that asserts on exact floating-point values, expect to widen a
tolerance. For one that asserts on a residual — `‖Ax − b‖` small — nothing
changes.

## When something looks wrong

Everything that is not accelerated is gonum's own code, so a wrong answer is
either in a routine this package overrides or in how it was called. To find out
which, take the backend out:

```go
blas64.Use(gonum.Implementation{})
```

If the problem survives that, it is not this package. If it disappears, the
[issue tracker](https://github.com/sebishogun/simdblas/issues) wants the
routine, the argument shapes and the sizes — the guards are shape-dependent, so
those three are usually enough to reproduce it.
