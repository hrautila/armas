
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Cholesky factorization

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_cholfactor) && defined(armas_x_cholfactor_w) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas1) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
#include "partition.h"
//! \endcond

static
int __unblk_cholfactor_lower(armas_x_dense_t *A, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a21, A22;
  int err = 0;
  DTYPE a11val;

  EMPTY(A00); EMPTY(a11);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  __nil,
                           __nil, &a21,  &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    // a21 = a21/a11
    a11val = armas_x_get(&a11, 0, 0);
    if (a11val > 0.0) {
      // a11 = sqrt(a11)
      a11val = __SQRT(a11val);
      armas_x_set(&a11, 0, 0, a11val);
      
      // a21 = a21/a11
      armas_x_invscale(&a21, a11val, conf);

      // A22 = A22 - a21*a21.T
      armas_x_mvupdate_sym(&A22, -__ONE, &a21, ARMAS_LOWER, conf);
    } else {
      if (err == 0) {
        conf->error = a11val < 0.0 ? ARMAS_ENEGATIVE : ARMAS_ESINGULAR;
        err = -1;
      }
    }

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }
  return err;
}

static
int __blk_cholfactor_lower(armas_x_dense_t *A, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A11, A21, A22;
  int err = 0;

  EMPTY(A00);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  __nil,
                           __nil, &A21,  &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    // A11 = CHOL(A11)
    if (__unblk_cholfactor_lower(&A11, conf) != 0) {
      err = err == 0 ? -1 : err;
    }
    // A21 = A21 * tril(A11).-T
    armas_x_solve_trm(&A21, __ONE, &A11, ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA, conf);

      // A22 = A22 - A21*A21.T
    armas_x_update_sym(__ONE, &A22, -__ONE, &A21, ARMAS_LOWER, conf);

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }

  // last block with unblocked
  if (ABR.rows > 0) {
    if (__unblk_cholfactor_lower(&ABR, conf) != 0) {
      err = err == 0 ? -1 : err;
    }
  }
  return err;
}


static
int __unblk_cholfactor_upper(armas_x_dense_t *A, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a12, A22;
  int err = 0;
  DTYPE a11val;

  EMPTY(A00); EMPTY(a11);

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    // a21 = a21/a11
    a11val = armas_x_get(&a11, 0, 0);
    if (a11val > 0.0) {
      // a11 = sqrt(a11)
      a11val = __SQRT(a11val);
      armas_x_set(&a11, 0, 0, a11val);
      
      // a12 = a12/a11
      armas_x_invscale(&a12, a11val, conf);

      // A22 = A22 - a12*a12.T
      armas_x_mvupdate_sym(&A22, -__ONE, &a12, ARMAS_UPPER, conf);
    } else {
      if (err == 0) {
        conf->error = a11val < 0.0 ? ARMAS_ENEGATIVE : ARMAS_ESINGULAR;
        err = -1;
      }
    }

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }
  return err;
}

static
int __blk_cholfactor_upper(armas_x_dense_t *A, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A11, A12, A22;
  int err = 0;

  EMPTY(A00); 
  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows -lb > 0 && ABR.cols - lb > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, __nil, &A22,  /**/  A, lb, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------
    // A11 = CHOL(A11)
    if (__unblk_cholfactor_upper(&A11, conf) != 0) {
      err = err == 0 ? -1 : err;
    }
    // A12 = tril(A11).-T * A12
    armas_x_solve_trm(&A12, __ONE, &A11, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSA, conf);

      // A22 = A22 - A12.T*A12
    armas_x_update_sym(__ONE, &A22, -__ONE, &A12, ARMAS_UPPER|ARMAS_TRANSA, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }

  if (ABR.rows > 0) {
    if (__unblk_cholfactor_upper(&ABR, conf) != 0) {
      err = err == 0 ? -1 : err;
    }
  }
  return err;
}

extern 
int __cholfactor_pv(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P,
                    int flags, armas_conf_t *conf);

extern 
int __cholsolve_pv(armas_x_dense_t *B, armas_x_dense_t *A, armas_pivot_t *P,
                    int flags, armas_conf_t *conf);

/**
 * \brief Cholesky factorization
 *
 * Compute the Cholesky factorization of a symmetric positive definite N-by-N matrix A.
 *
 * \param[in,out] A     
 *     On entry, the symmetric matrix A. If *ARMAS_UPPER* is set the upper triangular part
 *     of A contains the upper triangular part of the matrix A, and strictly
 *     lower part A is not referenced. If *ARMAS_LOWER* is set the lower triangular part
 *     of a contains the lower triangular part of the matrix A. Likewise, the
 *     strictly upper part of A is not referenced. On exit, factor U or L from the
 *     Cholesky factorization \f$ A = U^T U \f$ or \f$ A = L L^T \f$
 * \param[in] W
 *     Workspace for pivoting factorization.
 * \param[out] P
 *     Optional pivot array. If non null then pivoting factorization is computed. 
 * \param[in] flags 
 *      The matrix structure indicator, *ARMAS_UPPER* for upper tridiagonal and 
 *      *ARMAS_LOWER* for lower tridiagonal matrix.
 * \param[in,out] conf 
 *     Optional blocking configuration. If not provided default blocking configuration
 *     will be used.
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code
 *
 * Compatible with lapack.DPOTRF
 * \ingroup lapack
 */
int armas_x_cholfactor(armas_x_dense_t *A,
                       armas_pivot_t *P,
                       int flags,
                       armas_conf_t *conf)
{
  int err = 0;
  armas_wbuf_t wb = ARMAS_WBNULL;
  
  if (!conf)
    conf = armas_conf_default();

  if (!A) {
    conf->error = ARMAS_EINVAL;
    return -1;
  }
  if (P != ARMAS_NOPIVOT) {
    if (!armas_walloc(&wb, A->cols*sizeof(DTYPE))) {
      conf->error = ARMAS_EMEMORY;
      return -1;
    }
    err = armas_x_cholfactor_w(A, P, flags, &wb, conf);
    armas_wrelease(&wb);
    return err;
  }
  
  return armas_x_cholfactor_w(A, ARMAS_NOPIVOT, flags, ARMAS_NOWORK, conf); 
}

/**
 * \brief Cholesky factorization
 *
 * Compute the Cholesky factorization of a symmetric positive definite N-by-N matrix A.
 *
 */
int armas_x_cholesky(armas_x_dense_t *A,
                     int flags,
                     armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();

  if (!A) {
    conf->error = ARMAS_EINVAL;
    return -1;
  }
  return armas_x_cholfactor_w(A, ARMAS_NOPIVOT, flags, ARMAS_NOWORK, conf); 
}


/**
 * \brief Cholesky factorization
 *
 * Compute the Cholesky factorization of a symmetric positive definite N-by-N matrix A.
 *
 * \param[in,out] A     
 *     On entry, the symmetric matrix A. If *ARMAS_UPPER* is set the upper triangular part
 *     of A contains the upper triangular part of the matrix A, and strictly
 *     lower part A is not referenced. If *ARMAS_LOWER* is set the lower triangular part
 *     of a contains the lower triangular part of the matrix A. Likewise, the
 *     strictly upper part of A is not referenced. On exit, factor U or L from the
 *     Cholesky factorization \f$ A = U^T U \f$ or \f$ A = L L^T \f$
 * \param[out] P
 *     Optional pivot array. If non null then pivoting factorization is computed. 
 *     Set to ARMAS_NOPIVOT if normal cholesky factorization wanted.
 * \param[in] flags 
 *      The matrix structure indicator, *ARMAS_UPPER* for upper tridiagonal and 
 *      *ARMAS_LOWER* for lower tridiagonal matrix.
 * \param[in,out] wb
 *     Workspace for pivoting factorization. If wb.bytes is zero then work buffer size is
 *     returned in wb.bytes. 
 * \param[in,out] conf 
 *     Optional blocking configuration. If not provided default blocking configuration
 *     will be used.
 *
 * \retval  0 Success; If pivoting factorized then result matrix is full rank.
 * \retval >0 Success with pivoting factorization, matrix rank returned.
 * \retval -1 Error, `conf.error` holds error code
 *
 * Pivoting factorization is computed of P is not ARMAS_NOPIVOT. Pivoting factorization
 * needs workspace of size N elements for blocked version. If no workspace (ARMAS_NOWORK) is provided 
 * or it is too small then unblocked algorithm is used. 
 *
 * Factorization stops when diagonal element goes small enough.
 * Default value for stopping criteria is \$ max |diag(A)|*N*epsilon \$
 * If value of absolute stopping criteria `conf.stop` is non-zero then it is used. Otherwise
 * if `conf.smult` (relative stopping criterion multiplier) is non-zero then stopping criteria 
 * is set to \$ max |diag(A)|*smult \$.
 *
 * Pivoting factorization returns zero if result matrix is full rank. Return value greater than
 * zero is rank of result matrix. Negative values indicate error.
 *
 * 
 * Compatible with lapack.DPOTRF
 * \ingroup lapack
 */
int armas_x_cholfactor_w(armas_x_dense_t *A, 
                         armas_pivot_t *P,
                         int flags,
                         armas_wbuf_t *wb,
                         armas_conf_t *conf)
{
  armas_x_dense_t W;
  int err = 0;
  if (!conf)
    conf = armas_conf_default();

  if (!A) {
    conf->error = ARMAS_EINVAL;
    return -1;
  }

  if (P != ARMAS_NOPIVOT) {
    if (wb && wb->bytes == 0) {
      wb->bytes = A->cols * sizeof(DTYPE);
      return 0;
    }

    // working space is N elements for blocked factorization
    if (conf->lb > 0 && A->cols > conf->lb) {
      if (!wb || armas_wbytes(wb) < A->cols * sizeof(DTYPE))
        armas_x_make(&W, 0, 0, 0, (DTYPE *)0); // too small, force unblocked
      else
        armas_x_make(&W, A->cols, 1, A->cols, (DTYPE *)armas_wptr(wb));
    } else {
      // force unblocked with zero sized workspace
      armas_x_make(&W, 0, 0, 0, (DTYPE *)0);
    }
    return __cholfactor_pv(A, &W, P, flags, conf);
  }
  
  if (A->rows != A->cols) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }    

  if (conf->lb == 0 || A->cols <= conf->lb) {
    if (flags & ARMAS_LOWER) {
      err = __unblk_cholfactor_lower(A, conf);
    } else {
      err = __unblk_cholfactor_upper(A, conf);
    }
  } else {
    if (flags & ARMAS_LOWER) {
      err = __blk_cholfactor_lower(A, conf->lb, conf);
    } else {
      err = __blk_cholfactor_upper(A, conf->lb, conf);
    }
  }
  return err;
}

/**
 * \brief Solve symmetric positive definite system of linear equations 
 *
 * Solves a system of linear equations \f$ AX = B \f$ with symmetric positive
 * definite matrix A using the Cholesky factorization \f$ A = U^TU \f$ or \f$ A = LL^T \f$
 * computed by `cholfactor()`.
 *
 *  \param[in,out] B
 *      On entry, the right hand side matrix B. On exit, the solution matrix X.
 *  \param[in] A 
 *      The triangular factor U or L from Cholesky factorization as computed by
 *      `cholfactor().`
 * \param[in] P
 *     Optional pivot array. If non null then A is pivoted cholesky factorization.
 *     Set to ARMAS_NOPIVOT if normal cholesky factorization used.
 *  \param[in] flags 
 *      Indicator of which factor is stored in A. If *ARMAS_UPPER* (*ARMAS_LOWER) then upper
 *      (lower) triangle of A is stored. 
 *  \param[in,out] conf
 *      Optional blocking configuration. 
 *
 * \retval  0 Succes
 * \retval -1 Error, `conf.error` holds last error code
 *
 * Compatible with lapack.DPOTRS.
 * \ingroup lapack
 */
int armas_x_cholsolve(armas_x_dense_t *B,
                      const armas_x_dense_t *A, 
                      const armas_pivot_t *P,
                      int flags,
                      armas_conf_t *conf)
{
  int ok;
  if (!conf)
    conf = armas_conf_default();
  
  if (P != ARMAS_NOPIVOT) {
    return __cholsolve_pv(B, (armas_x_dense_t *)A, (armas_pivot_t *)P, flags, conf);
  }
  
  ok = B->rows == A->cols && A->rows == A->cols;
  if (!ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  if (flags & ARMAS_LOWER) {
    // solve A*X = B; X = A.-1*B == (L*L.T).-1*B == L.-T*(L.-1*B)
    armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_LOWER, conf);
    armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANSA, conf);
  } else {
    // solve A*X = B;  X = A.-1*B == (U.T*U).-1*B == U.-1*(U.-T*B)
    armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSA, conf);
    armas_x_solve_trm(B, __ONE, A, ARMAS_LEFT|ARMAS_UPPER, conf);
  }
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

