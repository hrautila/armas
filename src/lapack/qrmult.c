
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! QR orthogonal matrix multiplication

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qrmult) 
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

/*
 * Internal worksize calculation functions.
 */
static inline
int __ws_qrmult_left(int M, int N, int lb)
{
  return lb == 0 ? N : lb*(N+lb);
}

static inline
int __ws_qrmult_right(int M, int N, int lb)
{
  return lb == 0 ? M : lb*(M+lb);
}


/*
 * Unblocked algorith for computing C = Q.T*C and C = Q*C.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th column
 * below diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces H(k)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(k) == Q and C = Q*C
 */
static int
__unblk_qrmult_left(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22, *Aref;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;
  armas_x_dense_t CT, CB, C0, c1, C2;
  int pAdir, pAstart, pStart, pDir;
  int mb, nb, tb;

  EMPTY(A00); EMPTY(a11);

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = nb = tb = 0;
    Aref = &ABR;
  } else {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    tb = max(0, armas_x_size(tau) - A->cols);
    Aref = &ATL;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_2x1(&CT, 
                  &CB,   /**/  C,   mb, pStart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  armas_x_submatrix(&w12, W, 0, 0, C->cols, 1);

  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, pAdir);
    __repartition_2x1to3x1(&CT,
                           &C0,
                           &c1,
                           &C2,     /**/ C, 1, pDir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, pDir);
    // ---------------------------------------------------------------------------

    __apply_householder2x1(&t1, &a21, &c1, &C2, &w12, ARMAS_LEFT, conf);

    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, pAdir);
    __continue_3x1to2x1(&CT,
                        &CB,  /**/  &C0,  &c1,   C,   pDir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}


/*
 *
 */
static int
__blk_qrmult_left(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                  armas_x_dense_t *T, armas_x_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A10, A11, A20, A21, A22, AL, *Aref;
  armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  armas_x_dense_t CT, CB, C0, C1, C2;
  int pAdir, pAstart, pStart, pDir;
  int mb, nb, tb, transpose ;

  // initialize to GCC "maybe-uninitialized" error
  EMPTY(A00);

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = nb = tb = 0;
    Aref = &ABR;
    transpose = TRUE;
  } else {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    tb = max(0, armas_x_size(tau) - __IMIN(A->rows, A->cols));
    Aref = &ATL;
    transpose = FALSE;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_2x1(&CT, 
                  &CB,   /**/  C,   mb, pStart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &A10,  &A11,  __nil,
                           &A20,  &A21,  &A22,  /**/  A, lb, pAdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, A11.cols, pDir);
    __repartition_2x1to3x1(&CT,
                           &C0,
                           &C1,
                           &C2,     /**/ C, A11.cols, pDir);
    // ---------------------------------------------------------------------------
    // build block reflector
    __merge2x1(&AL, &A11, &A21);
    armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    armas_x_mscale(&Tcur, 0.0, ARMAS_NULL);
    __unblk_qr_reflector(&Tcur, &AL, &t1, conf);

    // compute Q*C or Q.T*C
    armas_x_submatrix(&Wrk, W, 0, 0, C1.cols, A11.cols);
    __update_qr_left(&C1, &C2, &A11, &A21, &Tcur, &Wrk, transpose, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, pAdir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
    __continue_3x1to2x1(&CT,
                        &CB,  /**/  &C0,  &C1,   C,   pDir);
  }
  return 0;
}


/*
 * Unblocked algorith for computing C = C*Q.T and C = C*Q.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th column
 * below diagonal in A.
 *
 *     Q.T = (H1(1)*H(2)*...*H(k)).T
 *         = H(k).T*...*H(2).T*H(1).T
 *         = H(k)...H(2)H(1)
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces C*H(1)H(2)...H(k) == C*Q.
 *
 * Progressing from bottom-right to top-left produces C*H(k)...H(2)H(1) == C*Q.T.
 */
static int
__unblk_qrmult_right(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                     armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22, *Aref;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;
  armas_x_dense_t CL, CR, C0, c1, C2;
  int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
  int mb, nb, tb, cb;

  EMPTY(A00); EMPTY(a11); 
  EMPTY(C0); EMPTY(CL);

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pCstart = ARMAS_PRIGHT;
    pCdir   = ARMAS_PLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->cols - A->cols);
    tb = max(0, armas_x_size(tau) - __IMIN(A->rows, A->cols));
    Aref = &ATL;
  } else {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pCstart = ARMAS_PLEFT;
    pCdir   = ARMAS_PRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = nb = tb = cb = 0;
    Aref = &ABR;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_1x2(&CL,   &CR,    /**/  C, cb, pCstart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  armas_x_submatrix(&w12, W, 0, 0, C->rows, 1);

  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, &a21,  &A22,  /**/  A, 1, pAdir);
    __repartition_1x2to1x3(&CL,
                           &C0,   &c1,   &C2,   /**/ C, 1, pCdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, pDir);
    // ---------------------------------------------------------------------------
    __apply_householder2x1(&t1, &a21, &c1, &C2, &w12, ARMAS_RIGHT, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, pAdir);
    __continue_1x3to1x2(&CL,   &CR,  /**/  &C0,  &c1,   C,   pCdir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}

/*
 * Blocked version for computing C = C*Q and C = C*Q.T from elementary reflectors
 * and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block reflector T.
 * Matrix C is updated by applying block reflector T using compact WY algorithm.
 */
static int
__blk_qrmult_right(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                   armas_x_dense_t *T, armas_x_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A10, A11, A20, A21, A22, AL, *Aref;
  armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  armas_x_dense_t CL, CR, C0, C1, C2;
  int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
  int mb, nb, cb, tb, transpose ;

  // initialize to "empty" to avoid "maybe-uninitialized" errors
  EMPTY(A00); EMPTY(A11); EMPTY(A22);
  EMPTY(CL); EMPTY(CR); EMPTY(C0);

  if (flags & ARMAS_TRANS) {
    // from bottom-right to top-left to produce transpose sequence (C*Q.T)
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    pCstart = ARMAS_PRIGHT;
    pCdir   = ARMAS_PLEFT;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->cols - A->cols);
    tb = max(0, armas_x_size(tau) - __IMIN(A->rows, A->cols));
    Aref = &ATL;
    transpose = TRUE;
  } else {
    // from top-left to bottom-right to produce normal sequence (C*Q)
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    pCstart = ARMAS_PLEFT;
    pCdir   = ARMAS_PRIGHT;
    mb = cb = tb = nb = 0;
    Aref = &ABR;
    transpose = FALSE;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,  /**/  A, mb, nb, pAstart);
  __partition_1x2(&CL, &CR,     /**/  C, cb, pCstart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &A10,  &A11,  __nil,
                           &A20,  &A21,  &A22,  /**/  A, lb, pAdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, A11.cols, pDir);
    __repartition_1x2to1x3(&CL,
                           &C0, &C1, &C2, /**/ C, A11.cols, pCdir);
    // ---------------------------------------------------------------------------
    // build block reflector
    __merge2x1(&AL, &A11, &A21);
    armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    armas_x_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_qr_reflector(&Tcur, &AL, &t1, conf);

    // compute Q*C or Q.T*C
    armas_x_submatrix(&Wrk, W, 0, 0, C1.rows, A11.cols);
    __update_qr_right(&C1, &C2, &A11, &A21, &Tcur, &Wrk, transpose, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, pAdir);
    __continue_1x3to1x2(&CL, &CR,    /**/  &C0,  &C1,   C,   pCdir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}

/**
 * \brief Multiply matrix with orthogonal matrix Q.
 *
 * Multiply and replace C with \f$ Q C \f$ or \f$ Q^T C \f$ where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    \f$ Q = H_1 H_2 . . . H_k \f$
 *
 * as returned by qrfactor().
 *
 * \param[in,out] C
 *   On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then N-by-M matrix
 *   On exit C is overwritten by \f$ Q C \f$ or \f$ Q^T C \f$. If bit *ARMAS_RIGHT* is 
 *   set then C is overwritten by \f$ C Q \f$ or \f$ CQ^T \f$.
 *
 * \param[in] A 
 *    QR factorization as returne by qrfactor() where the lower trapezoidal
 *    part holds the elementary reflectors.
 *
 * \param[in] tau
 *   The scalar factors of the elementary reflectors.
 *
 * \param[in,out] W
 *    Workspace matrix, required size is returned by qrmult_work().
 *
 * \param[in] flags 
 *   Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT*, *ARMAS_TRANS*
 *       
 * \param conf
 *   Blocking configuration. Field LB defines block size. If it is zero
 *   unblocked invocation is assumed. Actual blocking size is adjusted
 *   to available workspace size and minimum of configured block size and
 *   block size implied by workspace is used.
 *
 * Compatible with lapack.DORMQR
 * \ingroup lapack
 */
int armas_x_qrmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                   int flags, armas_conf_t *conf)
{
  WSSIZE wsizer;
  int wsmin, wsneed, lb, ok;
  armas_x_dense_t tauh;

  if (!conf)
    conf = armas_conf_default();

  // default to multiplication from left is nothing defined
  if (!(flags & (ARMAS_LEFT|ARMAS_RIGHT)))
    flags |= ARMAS_LEFT;

  if (flags & ARMAS_RIGHT) {
    ok = C->cols == A->rows;
    wsizer = __ws_qrmult_right;
  } else {
    ok = C->rows == A->rows;
    wsizer = __ws_qrmult_left;
  }

  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  lb = conf->lb;
  wsmin = wsizer(C->rows, C->cols, 0);
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = wsizer(C->rows, C->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(C->rows, C->cols, armas_x_size(W), wsizer);
    lb = __IMIN(lb, conf->lb);
  }

  EMPTY(tauh);
  armas_x_submatrix(&tauh, tau, 0, 0, A->cols, 1);
  if (lb == 0 || A->cols <= lb) {
    // unblocked 
    if (flags & ARMAS_LEFT) {
      __unblk_qrmult_left(C, A, tau, W, flags, conf);
    } else {
      __unblk_qrmult_right(C, A, tau, W, flags, conf);
    }
  } else {
    // blocked code
    armas_x_dense_t T, Wrk;

    // space for block reflector
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));

    if (flags & ARMAS_LEFT) {
      // temporary space after block reflector T, 
      armas_x_make(&Wrk, C->cols, lb, C->cols, &armas_x_data(W)[armas_x_size(&T)]);
      __blk_qrmult_left(C, A, &tauh, &T, &Wrk, flags, lb, conf);
    } else {
      // temporary space after block reflector T, 
      armas_x_make(&Wrk, C->rows, lb, C->rows, &armas_x_data(W)[armas_x_size(&T)]);
      __blk_qrmult_right(C, A, &tauh, &T, &Wrk, flags, lb, conf);
    }
  }
  return 0;
}

/**
 * \brief Calculate workspace for qrmult().
 *
 * Calculate required workspace with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 *
 * \param[in] A
 *    Matrix holding the elementary reflectors.
 * \param[in] flags
 *    Indicator flags
 * \param[in] conf
 *    Blocking configuration
 * \ingroup lapack
 */
int armas_x_qrmult_work(armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  if (flags & ARMAS_RIGHT) {
    return __ws_qrmult_right(A->rows, A->cols, conf->lb);
  }
  return __ws_qrmult_left(A->rows, A->cols, conf->lb);
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

