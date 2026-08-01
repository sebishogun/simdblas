package simdblas_test

import (
	"fmt"

	"github.com/sebishogun/simdblas"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// Installing the backend is the whole integration. Do it once, at startup,
// before anything numerical runs — every gonum package that reaches BLAS picks
// it up from there, including mat, stat and optimize.
func Example() {
	blas64.Use(simdblas.Implementation{})
	blas32.Use(simdblas.Implementation{})

	a := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	b := mat.NewDense(2, 2, []float64{5, 6, 7, 8})

	var c mat.Dense
	c.Mul(a, b)

	fmt.Println(mat.Formatted(&c))
	// Output:
	// ⎡19  22⎤
	// ⎣43  50⎦
}

// Matrix multiplication is where the largest gains are, because gemm has enough
// arithmetic per byte to keep a vector unit busy. Nothing about the calling
// code changes.
func ExampleImplementation_matrixMultiply() {
	blas64.Use(simdblas.Implementation{})

	a := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	b := mat.NewDense(3, 2, []float64{7, 8, 9, 10, 11, 12})

	var c mat.Dense
	c.Mul(a, b)

	fmt.Println(mat.Formatted(&c))
	// Output:
	// ⎡ 58   64⎤
	// ⎣139  154⎦
}

// Solving a linear system goes through an LU factorisation, which is built out
// of gemm and trsm rather than out of a single multiply. Both are accelerated,
// so this benefits too — less than a bare multiply does, because a
// factorisation also contains scalar work that no vector unit reaches.
func ExampleImplementation_solve() {
	blas64.Use(simdblas.Implementation{})

	//  4x +  y = 9
	//   x + 3y = 8
	a := mat.NewDense(2, 2, []float64{4, 1, 1, 3})
	b := mat.NewVecDense(2, []float64{9, 8})

	var x mat.VecDense
	if err := x.SolveVec(a, b); err != nil {
		fmt.Println("singular:", err)
		return
	}

	fmt.Printf("x = %.4f, y = %.4f\n", x.AtVec(0), x.AtVec(1))
	// Output: x = 1.7273, y = 2.0909
}

// Cholesky is the cheapest factorisation for a symmetric positive-definite
// matrix, and the one that benefits most here: it alternates trsm with syrk,
// and both are blocked through the accelerated gemm.
func ExampleImplementation_cholesky() {
	blas64.Use(simdblas.Implementation{})

	s := mat.NewSymDense(3, []float64{4, 1, 0, 1, 5, 2, 0, 2, 6})

	var chol mat.Cholesky
	if ok := chol.Factorize(s); !ok {
		fmt.Println("not positive definite")
		return
	}

	fmt.Printf("det = %.1f\n", chol.Det())
	// Output: det = 98.0
}

// The BLAS interface can also be used directly, without gonum's mat types. This
// is the same multiply as the first example, stated in the terms BLAS uses:
// row-major slices with an explicit leading dimension.
func ExampleImplementation_directBLAS() {
	var impl simdblas.Implementation

	a := []float64{1, 2, 3, 4} // 2x2
	b := []float64{5, 6, 7, 8} // 2x2
	c := make([]float64, 4)

	// C = 1*A*B + 0*C
	impl.Dgemm(blas.NoTrans, blas.NoTrans, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)

	fmt.Println(c)
	// Output: [19 22 43 50]
}

// float32 works the same way and needs its own registration: blas32 and blas64
// are separate globals, and installing one does nothing for the other.
func ExampleImplementation_float32() {
	blas32.Use(simdblas.Implementation{})

	var impl simdblas.Implementation
	x := []float32{1, 2, 3, 4}
	y := []float32{5, 6, 7, 8}

	fmt.Println(impl.Sdot(4, x, 1, y, 1))
	// Output: 70
}

// A triangular solve with several right-hand sides — the operation LAPACK
// spends half a factorisation in. Accelerated by blocking: gonum solves the
// small diagonal blocks, the fast gemm does the updates between them.
func ExampleImplementation_triangularSolve() {
	var impl simdblas.Implementation

	// Lower triangular, unit diagonal:
	//   [1 0]        [1]        [1  ]
	//   [2 1] * X =  [4]  ->  X=[2  ]
	a := []float64{1, 0, 2, 1}
	b := []float64{1, 4}

	impl.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit, 2, 1, 1, a, 2, b, 1)

	fmt.Println(b)
	// Output: [1 2]
}

// Results differ from gonum's in the last place, because no two BLAS
// implementations accumulate in the same order and the specification does not
// require them to. What this one guarantees instead is that the answer does not
// depend on the machine: the accumulation order is fixed rather than following
// the vector width, so a result computed here reproduces on any architecture.
func ExampleImplementation_reproducibility() {
	var impl simdblas.Implementation

	x := make([]float64, 4096)
	for i := range x {
		x[i] = 1.0 / float64(i+1)
	}

	// The same bits on amd64, arm64, riscv64 and the rest.
	fmt.Printf("%.10f\n", impl.Ddot(len(x), x, 1, x, 1))
	// Output: 1.6446899560
}
