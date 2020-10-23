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
   - Random Butterfly transformations

Currently tested only for single and double precision real numbers. 

### Building from sources

For compiling and installing from GitHub tree. Compiles with gcc 5.x and gcc 6.x.
```sh
  $ ./bootstrap.sh
  $ mkdir build
  $ cd build
  $ ../configure CFLAGS=-O3
  $ make
  $ sudo make install
```
To run some tests and see tests directory.
```sh
  $ make check
```

### Building documentation

Requires doxygen installed. Documentation created on `html` subdirectory.

```sh
$ cd build/docs
$ make
$ doxygen
```

###  Some numbers

For some performance numbers see file numbers.txt

### Using

Matrix interfaces provide a simple column major matrix implementation. See test programs
under `tests` subdirectory.

