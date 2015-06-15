## ARMAS - Another Rewrite of Matrix Algebra Subroutines


Simple column major matrix package and basic linear algebra subroutine (BLAS) implementation
with selected functionality from LAPACK linear algebra package. 

This library aims for single code base implementation of single and double precision real 
and complex versions of all supported functions. The implementation is inspired by libFLAME.

BLAS code supports 
 - level 1, 2 and 3 BLAS 
 - multi-threaded implementation of BLAS3 functions
 - SIMD accelerated for x86_64 
 - extended presicion versions of most BLAS functions. Implementation based on
   error free transformations as descripted by Ogita 2005.
 - configurable blocking parameters for BLAS3 functions

LAPACK functions
 - unblocked and blocked versions with configurable blocking factor
 - provides following factorization and solvers
   - QR family (QR, LQ, QL and RQ)
   - LU with partial pivoting
   - Cholesky
   - Bunch-Kaufman LDL factorization of symmetric matrices
   - SVD factorization
   - Symmetric EVD
 - other functionality
   - Hessenberg reduction
   - Bidiagonal reduction
   - Tridiagonal reduction
   - Givens rotations

Currently tested only for double precision real numbers. 

### Installation

From source package (not yet available). 
```sh
  $ ./configure
  $ make
  $ sudo make install
```
To run some tests
```sh
  $ make check
```
For compiling and installing from GitHub tree see INSTALLATION.

###  Some numbers

Performance numbers on few platforms. Programs in tests/perf directory.

**Lenovo S20**
    
    CPU :  Intel(R) Xeon(R) CPU W3550 @ 3.07GHz
    Test:  dgemm 4 CPUs
    N:  200,   7.4010,  10.9897,  12.9467 Gflops
    N:  400,  27.3908,  29.6872,  31.1132 Gflops
    N:  600,  14.0993,  20.3373,  31.4478 Gflops
    N:  800,  16.5798,  24.3145,  33.2931 Gflops
    N: 1000,  26.8118,  32.0073,  34.2912 Gflops
    N: 1200,  15.6206,  24.7353,  34.2304 Gflops
    N: 1400,  20.7980,  27.2520,  34.2317 Gflops
    N: 1600,  25.7324,  30.7854,  33.4283 Gflops

 Maximum theoretical performance of Nehalem CPU is 4 flops/cycle with SSE instructions. 
 With 4 cpus clocked @3.07Ghz we get 34.3/3.07/4 = 2.8 flops/cycle/cpu,
 equals ~ 70% of maximum.

 **Lenovo T530**
 
    CPU :  Intel(R) Core(TM) i7-3630QM CPU @ 2.40GHz
    Test:  dgemm 4 CPUs
    N:  200,  21.1338,  21.5806,  22.1630 Gflops
    N:  400,  28.0818,  45.0923,  54.3304 Gflops
    N:  600,  25.7633,  40.5718,  59.5621 Gflops
    N:  800,  61.8839,  62.1618,  62.4282 Gflops
    N: 1000,  63.0435,  63.2193,  63.3835 Gflops
    N: 1200,  49.4994,  59.6848,  62.9829 Gflops
    N: 1400,  64.2201,  64.4247,  64.5753 Gflops
    N: 1600,  48.1237,  57.2411,  63.8664 Gflops

 Maximum theoretical performance of Ivy Bridge CPU is 8 flops/cycle with AVX instructions. 
 With 4 cpus clocked @2.40Ghz we get 64.6/2.40/4 = 6.7 flops/cycle/cpu,
 equals ~ 84% of maximum.
 
### Using

Matrix interfaces provide a simple column major matrix implementation. 

#### Creating and deleting matrices

```c
  // declare matrix variable
  armas_d_dense_t A;
  armas_d_init(&A, N, N);
  ...
  // release storage
  armas_d_release(&A);
```
Same as above but with pointer variables.
```c
  armas_d_dense_t *A;
  A = armas_d_alloc(100, 100);
  ...
  armas_d_free(A);
```
Above versions allocate storage for matrix elements and assign ownership of the storage
area to initialized matrix variable. Separately allocated storage can also be used for matrix
storage. 
```c
  // on-stack buffer
  armas_d_dense_t A;
  double elems[20];
  // initialized A as 4x5 matrix with row stride of 4 
  armas_d_make(&A, 4, 5, 4, elems);
```  
Vectors are declared as matrices with one column or row.
```c
  // column vector
  armas_d_init(&x, 10, 1);
  // row vector
  armas_d_inti(&y, 1, 10);
```
#### Accessing elements

Single elements of a M-by-N matrix are accessed by [row,col] indexes where row = 0...M-1 and
col = 0...N-1. Negative indexes are accepted.
```c
  // get and set element at [i, j]
  val = armas_d_get(&A, i, j)
  armas_d_set(&A, i, j, val-1.0);

  // following are equivalent -- set bottom-right element to zero
  armas_d_set(&A, M-1, N-1, 0.0);
  armas_d_set(&A, -1, -1, 0.0);
```
Matrix views are supported. Matrix view (submatrix) shares storage with its parent matrix.
```c
  armas_d_init(&A, 50, 50);
  // B = A[10:30, 10:40]
  armas_d_submatrix(&B, &A, 10, 10, 20, 30);
  ...
  // setting element in parent, reading its value in submatrix
  armas_d_set(&A, 10, 10, 5.0);
  val = armas_d_get(&B, 0, 0);  // val == 5.0
```

Matrix rows and columns can be accessed as views. 
```c
  armas_d_init(&A, 10, 10);
  // row vector,    r.rows == 1, r.cols == 10, r.step == 10
  armas_d_row(&r, &A, 2);     
  // column vector, c.rows == 10, c.cols == 1, r.step == 10
  armas_d_column(&c, &A, 2);  
```

Matrix diagonals can be accessed as row vectors.
```c
  // D = diag(A)
  armas_d_diag(&D, &A, 0);
  // S = first superdiagonal of A
  armas_d_diag(&S, &A, 1);
  // E = first subdiagonal of A
  armas_d_diag(&E, &A, -1);
```

#### Setting and output

Setting values with value function as A[i,j] = func(i, j)
```c
  // all elements
  armas_d_set_values(&A, func, ARMAS_ANY);

  // set strictly lower triangular part of L
  armas_d_set_values(&L, func, ARMAS_LOWER|ARMAS_UNIT);

  // set upper triangular part of U
  armas_d_set_values(&U, func, ARMAS_UPPER);

  // set symmetric
  armas_d_set_values(&S, func, ARMAS_SYMM);
```

Printing matrix to stream.
```c
  armas_d_printf(stdout, "%7.4f", &A);
```

#### Basic operations

Vector and matrix variables in capital letters, type is pointer to matrix ie. armas_d_dense_t*.

 
Copying, B = A
```c
  armas_d_mcopy(B, A);
```
Transpose, B = A.T
```c
  armas_d_transpose(B, A);
```
Element-wise scaling, B = alpha*B
```c
  armas_d_mscale(B, alpha);
```
Element-wise adding with constant, B += alpha
```c
  armas_d_madd(B, alpha)
```
#### Basic linear algebra routines

Configuration block holds tuning parameters for low level kernel functions and error code of the
last error. Error code is not cleared at function entry. 

##### Vector-vector operations (Blas level 1)

Dot product of two vectors (blas.DDOT)
```c
   value = armas_d_dot(X, Y, conf)
```
Vector 2-norm (blas.DNRM2)
```c
   value = armas_d_nrm2(X, conf)
```
Sum of absolute values of vector elements (blas.DASUM)
```c
   asum = armas_d_asum(X, conf);
```
Sum of vector elements
```c
   sum = armas_d_sum(X, conf)
```
Index of maximum absolute value (blas.DIAMAX)
```c
   index = armas_d_iamax(X, conf)
```
Vector-vector sum Y = Y + alpha*X (blas.DAXPY)
```c
   armas_d_axpy(Y, X, alpha, conf)
```
Extended vector-vector sum Y = beta*Y + alpha*X 
```c
   armas_d_axpby(Y, X, alpha, beta, conf)
```

##### Vector-matrix operations (BLAS level 2)

Flag bits in operations:

    ARMAS_TRANS  : matrix is transposed
    ARMAS_UPPER  : matrix is upper triangular
    ARMAS_LOWER  : matrix is lower trinngular
    ARMAS_UNIT   : matrix is unit diagonal

General matrix-vector product Y = beta*Y + alpha*op(A)*X (blas.DGEMV)
```c
   armas_d_mvmult(Y, A, X, alpha, beta, flags, conf)
```
Symmetric matrix-vector product Y = beta*Y + alpha*op(A)*X (blas.DSYMV)
```c
   armas_d_mvmult_sym(Y, A, X, alpha, beta, flags, conf)
```
Triangular matrix-vector product Y = alpha*op(A)*Y (blas.DTRMV)
```c
   armas_d_mvmult_trm(Y, A, alpha, flags, conf)
```
Triangular matrix-vector solve Y = alpha*op(A.-1)*Y (blas.DTRSV)
```c
   armas_d_mvsolve_trm(Y, A, alpha, flags, conf)
```
General matrix rank update  A = A + alpha*X*Y  (blas.GER)
```c
   armas_d_mvupdate(A, X, Y, alpha, conf)
```
Symmetric matrix rank update  A = A + alpha*X*X.T  (blas.SYR)
```c
   armas_d_mvupdate_sym(A, X, alpha, flags, conf)
```
Symmetric matrix rank-2 update  A = A + alpha*X*Y.T + alpha*Y*X.T  (blas.SYR2)
```c
   armas_d_mvupdate2_sym(A, X, Y, alpha, flags, conf)
```

##### Matrix-matrix operations (BLAS level 3)

Additional flag bits in operations:

    ARMAS_TRANSA : first matrix is transposed
    ARMAS_TRANSB : second matrix is transposed
    ARMAS_LEFT   : operations from left
    ARMAS_RIGHT  : operations from right

General matrix-matrix product C = beta*C + alpha*op(A)*op(B) (blas.DGEMM)
```c
   armas_d_mult(C, A, B, alpha, beta, flags, conf)
```
Symmetric matrix-matrix product C = beta*C + alpha*op(A)*op(B) (blas.DSYMM)
```c
   armas_d_mult_sym(C, A, B, alpha, beta, flags, conf)
```
Triangular matrix-matrix product B = alpha*op(A)*B (blas.DTRMM)
```c
   armas_d_mult_trm(B, A, alpha, flags, conf)
```
Triangular matrix-matrix solve B = alpha*op(A.-1)*B (blas.DTRSM)
```c
   armas_d_solve_trm(Y, A, alpha, flags, conf)
```
Symmetric matrix rank-k update  A = A + alpha*X*X.T  (blas.SYRK)
```c
   armas_d_update_sym(A, X, alpha, flags, conf)
```
Symmetric matrix rank-2k update  A = A + alpha*X*Y.T + alpha*Y*X.T  (blas.SYRK2)
```c
   armas_d_update2_sym(A, X, Y, alpha, flags, conf)
```

##### Linear algebra

New flag bits for linear algebra operations

    ARMAS_MULTQ  : multiply with Q
    ARMAS_MULTP  : multiply with Q

Cholesky factorization of positive definite matrix A = L*L.T  (lapack.DPOTRF)
```c
   armas_d_cholfactor(A, conf)
```
Solving of Cholesky factorized linear system  B = A.-1*B    (lapack.DPOTRS)
```c
   armas_d_cholsolve(B, A, conf)
```
LU factorization with partial pivoting of general matrix  A = P.-1*L*U*P  (lapack.DGETRF)
```c
   armas_d_lufactor(A, pivots, conf)
```
Solving of LU factorized linear system  (lapack.DGETRS)
```c
   armas_d_lusolve(B, A, pivots, conf)
```
Bunch-Kauffman LDL.T factorization of symmetric real matrix (lapack.DSYTRF)
```c
   armas_d_bkfactor(A, W, pivots, flags, conf)
```
Solving of BK factorized linear system (lapackd.DSYTRS)
```c
   armas_d_bksolve(B, A, W, pivots, flags, conf)
```
QR factorization of general M-by-N matrix A = Q*R (lapack.DGEQRF)
```c
   armas_d_qrfactor(A, tau, W, conf)
```
Building the orthogonal matrix Q from QR factorization (lapack.DORGQR)
```c
   armas_d_qrbuild(A, tau, W, conf)
```
Multiplying general matrix with orthogonal matrix Q from QR factorization (lapack.DORMQR)
```c
   armas_d_qrmult(C, A, tau, W, flags, conf)
```
Solving of QR factorized linear system
```c
   armas_d_qrsolve(B, A, tau, W, flags, conf)
```
LQ factorization of general M-by-N matrix A = L*Q (lapack.DGELQF)
```c
   armas_d_lqfactor(A, tau, W, conf)
```
Building the orthogonal matrix Q from LQ factorization (lapack.DORGLQ)
```c
   armas_d_lqbuild(A, tau, W, conf)
```
Multiplying general matrix with orthogonal matrix Q from LQ factorization (lapack.DORMLQ)
```c
   armas_d_lqmult(C, A, tau, W, flags, conf)
```
Solving of LQ factorized linear system
```c
   armas_d_lqsolve(B, A, tau, W, flags, conf)
```
QL factorization of general M-by-N matrix A = Q*L (lapack.DGEQLF)
```c
   armas_d_qlfactor(A, tau, W, conf)
```
Building the orthogonal matrix Q from QL factorization (lapack.DORGQL)
```c
   armas_d_qlbuild(A, tau, W, conf)
```
Multiplying general matrix with orthogonal matrix Q from QL factorization (lapack.DORMQL)
```c
   armas_d_qlmult(C, A, tau, W, flags, conf)
```
Hessenberg reduction of general N-by-N matrix A = H*B*H.T (lapackd.DGEHRD)
```c
  armas_d_hessreduce(A, tau, W, conf)
```
Multiplying general matrix with orthogonal matrix Q from Hessenberg reduction (lapack.DORMHR)
```c
   armas_d_hessmult(C, A, tau, W, flags, conf)
```
Bidiagonalization of general M-by-N matrix A = Q*B*P.T (lapackd.DGEBRD)
```c
  armas_d_bdreduce(A, tauq, taup, W, conf)
```
Multiplying general matrix with orthogonal matrix Q or P from Bidiagonal reduction (lapack.DORMBR)
```c
   armas_d_bdmult(C, A, tau, W, flags, conf)
```
Tridiagonalization of symmetric N-by-N matrix A = Q*T*Q.T (lapackd.DSYTRD)
```c
  armas_d_trdreduce(A, tau, W, flags, conf)
```
Multiplying general matrix with orthogonal matrix Q from tridiagonal reduction (lapack.DORMTR)
```c
   armas_d_trdmult(C, A, tau, W, flags, conf)
```
Eigenvalues and eigenvectors of tridiagonal symmetric matrix (lapack.DSTEQR)
```c
   armas_d_trdeigen(D, E, V, A, W, flags, conf)
```
Eigenvalues and eigenvectors of symmetric matrix (lapack.DSYEV)
```c
   armas_d_eigen_sym(D, A, W, flags, conf)
```
Singular values of bidiagonal matrix (lapack.DBDSQR)
```c
   armas_d_bdsvd(D, E, U, V, A, W, flags, conf)
```
Singular values and singular vectors of general matrix (lapack.GESVD)
```c
   armas_d_svd(S, U, V, A, W, flags, conf)
```
Functions for computing and applying Givens rotations
```c
   armas_d_gvcompute(&cos, &sin, &r, a, b)
   armas_d_gvrotate(&v0, &v1, cos, sin, y0, y1)
   armas_d_gvleft(A, cos, sin, row1, row2, col, ncol)
   armas_d_gvright(A, cos, sin, col1, col2, row, nrow)
```


