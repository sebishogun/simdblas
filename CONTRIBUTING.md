# Contributing

## The bar for a change

Every routine here replaces one that already works. So the question is never
"is this faster" on its own — it is "is this faster, *and* does it return an
answer gonum would recognise, *and* does it hand the call back when it is not
faster".

All three are tested. A change that misses the third is the one that will not
be noticed, because nothing fails: the numbers are right and the program is
slower than it was before anyone installed this.

## Building and testing

```
go test ./...          # differential against gonum, every routine
go test -race ./...    # gemm and gemv run concurrently above a threshold
```

The test suite compares each override against the gonum routine it replaces at
two bars. Where the fast path does **not** apply this package delegates, so the
answer must be gonum's *exactly, to the bit* — anything else means a guard is
wrong and a fast path ran when it should not have. Where it does apply, the bar
is a relative tolerance, because the accumulation orders differ legitimately.

## Adding an accelerated routine

1. **Override the method.** `Implementation` embeds `gonum.Implementation`, so
   the routine already exists and works. You are replacing a working method,
   not writing a new one.
2. **Guard the shape.** Check what your fast path requires and delegate
   everything else, invalid arguments included — that is what keeps the panics
   identical to gonum's.
3. **Measure the crossover** and add a minimum size. Nearly every routine has
   one. Set it at the first size that is clearly ahead, not at break-even:
   being wrong toward gonum costs a little speed, being wrong the other way is
   a regression a caller will notice.
4. **Test both paths.** Exact equality below the threshold, tolerance above it.
5. **Test that the fast path runs.** A suite that passes by delegating
   everything looks exactly like one that works. See
   `TestFastPathActuallyRuns`, which requires `Ddot` and `Dnrm2` *not* to
   return gonum's exact bits.

## Benchmarking

Both implementations in the same process on the same inputs; anything else
compares two builds. `-count 6` or more, take the minimum rather than the mean,
and run the comparison twice — benchmark noise is one-sided, so the fastest run
is the one least interfered with. A number from a busy machine is worse than no
number: an earlier draft of the README's table was measured against a container
using every core and reported 4.81× as 2.27×; re-measured for v1.0.1, the
quiet value is 4.2×.

## Reporting a run on another architecture

**This is the contribution the project most needs.** Everything in the README
was measured on one amd64 machine. simd.go underneath is verified on amd64 and
arm64 NEON and under emulation elsewhere, but this package has never been
timed anywhere but here — and the thresholds are measured constants, so a
machine with a different cache or a narrower vector unit may well want
different ones.

Two commands:

```
go test ./...
go test -run '^$' -bench . -count 6 .
```

Open an issue with the output and `go env GOARCH`, or a pull request. A failing
run is more useful than a passing one; do not clean it up first.

## Style

Match the surrounding code. Comment density is high on purpose: the comments
explain why the obvious thing is wrong, which is what does not survive in a
diff. A threshold without the measurement beside it is a magic number, and the
next person to touch it has to redo the work to know whether it is still right.
