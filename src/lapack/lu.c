
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_lufactor) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_blas2) && defined(__armas_blas3)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

static
int __unblk_lufactor_nopiv(__armas_dense_t *A, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
  DTYPE a11val;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    // a21 = a21/a11
    a11val = __armas_get(&a11, 0, 0);
    __armas_invscale(&a21, a11val, conf);

    // A22 = A22 - a21*a12
    __armas_mvupdate(&A22, &a21, &a12, -1.0, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }
  return 0;
}

static
int __blk_lufactor_nopiv(__armas_dense_t *A, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A21, A22;
  DTYPE a11val;

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    // A11 = LU(A11)
    __unblk_lufactor_nopiv(&A11, conf);
    // A12 = trilu(A11)*A12.-1
    __armas_solve_trm(&A12, &A11, 1.0, ARMAS_LEFT|ARMAS_LOWER|ARMAS_UNIT, conf);
    // A21 = A21.-1*triu(A11)
    __armas_solve_trm(&A21, &A11, 1.0, ARMAS_RIGHT|ARMAS_UPPER, conf);
    // A22 = A22 - A21*A12
    __armas_mult(&A22, &A21, &A12, -1.0, 1.0, ARMAS_NONE, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }

  // last block with unblocked 
  if (ABR.rows > 0 && ABR.cols > 0) {
    __unblk_lufactor_nopiv(&ABR, conf);
  }
  return 0;
}

/*
 * Unblocked factorization with partial (row) pivoting. 'Left looking' version.
 */
static
int __unblk_lufactor(__armas_dense_t *A, armas_pivot_t *P, int offset, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, a01, a10, a11, a21, A20, A22;
  __armas_dense_t AL, AR, A0, a1, A2, aB0;
  armas_pivot_t pT, pB, p0, p1, p2;
  int pi, err = 0;
  DTYPE a11val, aa;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_1x2(&AL,  &AR,     /**/  A, 0, ARMAS_PLEFT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PTOP);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00, &a01, __nil,
                           &a10, &a11, __nil,
                           &A20, &a21, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    __repartition_1x2to1x3(&AL,
                           &A0, &a1,  &A2,      /**/  A, 1, ARMAS_PRIGHT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, 1, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    // A. apply previous pivots and updates to current column   
    __apply_pivots(&a1, &p0, conf);
    // a. a01 = trilu(A00)\a01
    __armas_mvsolve_trm(&a01, &A00, 1.0, ARMAS_LEFT|ARMAS_LOWER|ARMAS_UNIT, conf);
    // b. a11 = a11 - a10*a01
    a11val  = __armas_get(&a11, 0, 0);
    aa      = __armas_dot(&a10, &a01, conf);
    __armas_set(&a11, 0, 0, a11val - aa); 
    // c. a21 = a21 - A20*a01
    err = __armas_mvmult(&a21, &A20, &a01, -1.0, 1.0, ARMAS_NONE, conf);
    // HERE: current column has been updated with effects of earlier computations.

    // B. find pivot index on vector ( a11 ) == ABR[:,0] (1st column of ABR)
    //                               ( a21 )
    __armas_column(&aB0, &ABR, 0);
    pi = __pivot_index(&aB0, conf);
    armas_pivot_set(&p1, 0, pi);
    __apply_pivots(&aB0, &p1, conf);

    // a21 = a21 / a11
    a11val = __armas_get(&a11, 0, 0);
    if (a11val == 0.0) {
      conf->error = ARMAS_ESINGULAR;
      err = -1;
    } else {
      __armas_invscale(&a21, a11val, conf);
    }
    // apply pivots on left columns
    __apply_pivots(&ABL, &p1, conf);
    // pivot index to origin of matrix row numbers
    armas_pivot_set(&p1, 0, pi+ATL.rows);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR,  /**/  &A00, &a11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_1x3to1x2(&AL,   &AR,  /**/  &A0 , &a1, /**/  A, ARMAS_PRIGHT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
  }

  if (ABR.cols > 0) {
    // here A.rows < A.cols; handle the right columns
    __apply_pivots(&ATR, P, conf);
    __armas_solve_trm(&ATR, &ATL, 1.0, ARMAS_LEFT|ARMAS_UNIT|ARMAS_LOWER, conf);
  }
  return err;
}


/*
 * Blocked LU factorization with row pivoting. 'Left looking' version, pivots
 * updated on current block and left blocks.
 */
static
int __blk_lufactor(__armas_dense_t *A, armas_pivot_t *P, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ATR, ABL, ABR, A00, A01, A10, A11, A21, A20, A22;
  __armas_dense_t AL, AR, A0, A1, A2, AB0;
  armas_pivot_t pT, pB, p0, p1, p2;
  int k, pi, err = 0;

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);
  __partition_1x2(&AL,  &AR,    /**/  A, 0, ARMAS_PLEFT);
  __pivot_2x1(&pT,
              &pB,   /**/  P, 0, ARMAS_PTOP);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00, &A01, __nil,
                           &A10, &A11, __nil,
                           &A20, &A21, &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    __repartition_1x2to1x3(&AL,
                           &A0, &A1,  &A2,    /**/  A, A11.cols, ARMAS_PRIGHT);
    __pivot_repart_2x1to3x1(&pT,
                            &p0,
                            &p1,
                            &p2,     /**/ P, A11.cols, ARMAS_PBOTTOM);
    // ---------------------------------------------------------------------------
    // A. apply previous pivots and updates to current column   
    __apply_pivots(&A1, &p0, conf);
    // a. A01 = trilu(A00) \ A01
    __armas_solve_trm(&A01, &A00, 1.0, ARMAS_LEFT|ARMAS_LOWER|ARMAS_UNIT, conf);
    // b. A11 = A11 - A10*A01
    __armas_mult(&A11, &A10, &A01, -1.0, 1.0, ARMAS_NONE, conf);
    // c. A21 = A21 - A20*A01
    __armas_mult(&A21, &A20, &A01, -1.0, 1.0, ARMAS_NONE, conf);
    // HERE: current block has been updated with effects of earlier computations.

    // B. factor ( A11 ) 
    //           ( A21 )      
    __merge2x1(&AB0, &A11, &A21);
    err = __unblk_lufactor(&AB0, &p1, ATL.rows, conf);

    // apply pivots on left columns
    __apply_pivots(&ABL, &p1, conf);

    // shift pivot indicies to origin of matrix row numbers
    for (k = 0; k < armas_pivot_size(&p1); k++) {
      pi = armas_pivot_get(&p1, k);
      armas_pivot_set(&p1, k, pi+ATL.rows);
    }
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR,  /**/  &A00, &A11, &A22, /**/  A, ARMAS_PBOTTOMRIGHT);
    __continue_1x3to1x2(&AL,   &AR,  /**/  &A0 , &A1, /**/  A, ARMAS_PRIGHT);
    __pivot_cont_3x1to2x1(&pT,
                          &pB, /**/ &p0, &p1, /**/ P, ARMAS_PBOTTOM);
  }

  if (ABR.cols > 0) {
    // here A.rows < A.cols; handle the right columns
    __apply_pivots(&ATR, P, conf);
    __armas_solve_trm(&ATR, &ATL, 1.0, ARMAS_LEFT|ARMAS_UNIT|ARMAS_LOWER, conf);
  }
  return err;
}


/*
 * Compute an LU factorization of a general M-by-N matrix using
 * partial pivoting with row interchanges.
 *
 * Arguments:
 *   A      On entry, the M-by-N matrix to be factored. On exit the factors
 *          L and U from factorization A = P*L*U, the unit diagonal elements
 *          of L are not stored.
 *
 *   pivots On exit the pivot indices. 
 *
 *   conf   Blocking configuration.
 *
 * Compatible with lapack.DGETRF
 */
int __armas_lufactor(__armas_dense_t *A, armas_pivot_t *P, armas_conf_t *conf)
{
  int lb, err = 0;
  if (!conf)
    conf = armas_conf_default();

  lb = conf->lb;
  if (lb == 0 || A->cols <= lb || A->rows <= lb) {
    if (P) {
      err = __unblk_lufactor(A, P, 0, conf);
    } else {
      err = __unblk_lufactor_nopiv(A, conf);
    }
  } else {
    if (P) {
      err = __blk_lufactor(A, P, lb, conf);
    } else {
      err = __blk_lufactor_nopiv(A, lb, conf);
    }
  }
  return err;
}

/*
 * Solve a system of linear equations A*X = B or A.T*X = B with general N-by-N
 * matrix A using the LU factorization computed by armas_lufactor().
 *
 * Arguments:
 *  B      On entry, the right hand side matrix B. On exit, the solution matrix X.
 *
 *  A      The factor L and U from the factorization A = P*L*U as computed by
 *         armas_lufactor()
 *
 *  pivots The pivot indices from armas_lufactor(), if NULL then no pivoting applied
 *         to matrix B.
 *
 *  flags  The indicator of the form of the system of equations.
 *         If flags&TRANSA then system is transposed. 
 *
 * Compatible with lapack.DGETRS.
 */
int __armas_lusolve(__armas_dense_t *B, __armas_dense_t *A,
                    armas_pivot_t *P, int flags, armas_conf_t *conf)
{
  int lb, ok;
  if (!conf)
    conf = armas_conf_default();
  
  ok = B->rows == A->cols && A->rows == A->cols;
  if (!ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }
  if (P) {
    // pivots; apply to B.
    __apply_row_pivots(B, P, PIVOT_FORWARD, conf);
  }

  if (flags & ARMAS_TRANS) {
    // solve A.T*X = B; X = A.-T*B == (L.T*U.T).-1*B == U.-T*(L.-T*B)
    __armas_solve_trm(B, A, 1.0, ARMAS_LEFT|ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANSA, conf);
    __armas_solve_trm(B, A, 1.0, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSA, conf);
  } else {
    // solve A*X = B;  X = A.-1*B == (L*U).-1*B == U.-1*(L.-1*B)
    __armas_solve_trm(B, A, 1.0, ARMAS_LEFT|ARMAS_LOWER|ARMAS_UNIT, conf);
    __armas_solve_trm(B, A, 1.0, ARMAS_LEFT|ARMAS_UPPER, conf);
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

