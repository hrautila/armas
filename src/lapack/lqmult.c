
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_lqmult) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

/*
 * Internal worksize calculation functions.
 */
static inline
int __ws_lqmult_left(__armas_dense_t *A, int lb)
{
  return lb == 0 ? A->cols : lb*(A->cols+lb);
}

static inline
int __ws_lqmult_right(__armas_dense_t *A, int lb)
{
  return lb == 0 ? A->rows : lb*(A->rows+lb);
}

/*
 * Unblocked algorith for computing C = Q.T*C and C = Q*C.
 *
 * Q = H(k)H(k-1)...H(1) where elementary reflectors H(i) are stored on i'th row
 * right of diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller row numbers
 * to larger, produces H(k)...H(2)H(1) == Q. and C = Q*C
 *
 * Progressing from bottom-right to top-left produces H(k)H(k-1)...H(1) == Q.T and C = Q.T*C
 */
static int
__unblk_lqmult_left(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                    __armas_dense_t *W, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, a21, A22, *Aref;
  __armas_dense_t tT, tB, t0, t1, t2, w12;
  __armas_dense_t CT, CB, C0, c1, C2;
  int pAdir, pAstart, pStart, pDir;
  int mb, nb, tb, cb;
  int nerr, k;

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->rows - A->rows);
    tb = max(0, __armas_size(tau) - A->rows);
    Aref = &ATL;
  } else {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = nb = tb = cb = 0;
    Aref = &ABR;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_2x1(&CT, 
                  &CB,   /**/  C,   cb, pStart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  __armas_submatrix(&w12, W, 0, 0, C->cols, 1);

  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, pAdir);
    __repartition_2x1to3x1(&CT,
                           &C0,
                           &c1,
                           &C2,     /**/ C, 1, pDir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, pDir);
    // ---------------------------------------------------------------------------
    __apply_householder2x1(&t1, &a12, &c1, &C2, &w12, ARMAS_LEFT, conf);
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

static int
__blk_lqmult_left(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                  __armas_dense_t *T, __armas_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A22, AR, *Aref;
  __armas_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  __armas_dense_t CT, CB, C0, C1, C2;
  int pAdir, pAstart, pStart, pDir;
  int mb, nb, tb, cb, transpose ;

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->rows - A->rows);
    tb = max(0, __armas_size(tau) - min(A->rows, A->cols));
    Aref = &ATL;
    transpose = FALSE;
  } else {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = nb = tb = cb = 0;
    Aref = &ABR;
    transpose = TRUE;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_2x1(&CT, 
                  &CB,   /**/  C,   cb, pStart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &A11,  &A12,
                           __nil, __nil, &A22,  /**/  A, lb, pAdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, pDir);
    __repartition_2x1to3x1(&CT,
                           &C0,
                           &C1,
                           &C2,     /**/ C, A11.cols, pDir);
    // ---------------------------------------------------------------------------
    // build block reflector
    __merge1x2(&AR, &A11, &A12);
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __armas_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_lq_reflector(&Tcur, &AR, &t1, conf);

    // compute Q*C or Q.T*C
    __armas_submatrix(&Wrk, W, 0, 0, C1.cols, A11.cols);
    __update_lq_left(&C1, &C2, &A11, &A12, &Tcur, &Wrk, transpose, conf);
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
 * Q = H(k)H(k-1)...H(1) where elementary reflectors H(i) are stored on i'th row
 * right of diagonal in A.
 *
 *     Q.T = (H1(k)*...H(2)*H(1)).T
 *         = H(1).T*H(2)*T...*H(1).T
 *         = H(1)H(2)...H(k)
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces C*H(1)H(2)...H(k) == C*Q.T.
 *
 * Progressing from bottom-right to top-left produces C*H(k)...H(2)H(1) == C*Q.
 */
static int
__unblk_lqmult_right(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                     __armas_dense_t *W, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a11, a12, A22, *Aref;
  __armas_dense_t tT, tB, t0, t1, t2, w12;
  __armas_dense_t CL, CR, C0, c1, C2;
  int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
  int mb, nb, tb, cb;

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pCstart = ARMAS_PLEFT;
    pCdir   = ARMAS_PRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = nb = tb = cb = 0;
    Aref = &ABR;
  } else {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pCstart = ARMAS_PRIGHT;
    pCdir   = ARMAS_PLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->cols - A->rows);
    tb = max(0, __armas_size(tau) - min(A->rows, A->cols));
    Aref = &ATL;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_1x2(&CL,   &CR,    /**/  C, cb, pCstart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  __armas_submatrix(&w12, W, 0, 0, C->rows, 1);

  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           __nil, &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, pAdir);
    __repartition_1x2to1x3(&CL,
                           &C0,   &c1,   &C2,   /**/ C, 1, pCdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, pDir);
    // ---------------------------------------------------------------------------
    __apply_householder2x1(&t1, &a12, &c1, &C2, &w12, ARMAS_RIGHT, conf);
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
__blk_lqmult_right(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                   __armas_dense_t *T, __armas_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A11, A12, A21, A22, AR, *Aref;
  __armas_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  __armas_dense_t CL, CR, C0, C1, C2;
  int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
  int mb, nb, cb, tb, transpose ;

  if (flags & ARMAS_TRANS) {
    // from top-left to bottom-right to produce normal sequence (C*Q.T)
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    pCstart = ARMAS_PLEFT;
    pCdir   = ARMAS_PRIGHT;
    mb = cb = tb = nb = 0;
    Aref = &ABR;
    transpose = TRUE;
  } else {
    // from bottom-right to top-left to produce transpose sequence (C*Q)
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    pCstart = ARMAS_PRIGHT;
    pCdir   = ARMAS_PLEFT;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->cols - A->rows);
    tb = max(0, __armas_size(tau) - min(A->rows, A->cols));
    Aref = &ATL;
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
                           __nil, &A11,  &A12,
                           __nil, &A21,  &A22,  /**/  A, lb, pAdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, pDir);
    __repartition_1x2to1x3(&CL,
                           &C0, &C1, &C2, /**/ C, A11.cols, pCdir);
    // ---------------------------------------------------------------------------
    // build block reflector
    __merge1x2(&AR, &A11, &A12);
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __armas_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_lq_reflector(&Tcur, &AR, &t1, conf);

    // compute Q*C or Q.T*C
    __armas_submatrix(&Wrk, W, 0, 0, C1.rows, A11.cols);
    __update_lq_right(&C1, &C2, &A11, &A12, &Tcur, &Wrk, transpose, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, pAdir);
    __continue_1x3to1x2(&CL, &CR,    /**/  &C0,  &C1,   C,   pCdir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}


/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(k)H(k-1)...H(1)
 *
 * as returned by DecomposeLQ().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then
 *        N-by-M matrix.  On exit C is overwritten by Q*C or Q.T*C.
 *        If bit RIGHT is set then C is  overwritten by C*Q or C*Q.T
 *
 *  A     LQ factorization as returne by lqfactor() where the upper
 *        trapezoidal part holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors.
 *
 *  W     Workspace matrix,  required size is returned by WorksizeMultQ().
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block sized. If it is zero
 *        unblocked invocation is assumed.
 *
 * Compatible with lapack.DORMLQ
 *
 * Notes:
 *   m(A) is number of elementary reflectors == A.rows
 *   n(A) is the order of the Q matrix == A.cols
 *
 *   LEFT : m(C) == n(A)
 *   RIGHT: n(C) == n(A)
 */
int __armas_lqmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                   int flags, armas_conf_t *conf)
{
  WSFUNC wsizer;
  int wsmin, lb, ok;
  if (!conf)
    conf = armas_conf_default();

  if (flags & ARMAS_RIGHT) {
    ok = C->cols == A->cols;
    wsizer = __ws_lqmult_right;
  } else {
    ok = C->rows == A->cols;
    wsizer = __ws_lqmult_left;
  }

  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  lb = conf->lb;
  wsmin = wsizer(C, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }

  if (lb == 0 || A->cols <= lb) {
    // unblocked 
    if (flags & ARMAS_LEFT) {
      __unblk_lqmult_left(C, A, tau, W, flags, conf);
    } else {
      __unblk_lqmult_right(C, A, tau, W, flags, conf);
    }
  } else {
    // blocked code
    __armas_dense_t T, Wrk;

    // space for block reflector
    __armas_make(&T, lb, lb, lb, __armas_data(W));

    if (flags & ARMAS_LEFT) {
      // temporary space after block reflector T, 
      __armas_make(&Wrk, C->cols, lb, C->cols, &__armas_data(W)[__armas_size(&T)]);
      __blk_lqmult_left(C, A, tau, &T, &Wrk, flags, lb, conf);
    } else {
      // temporary space after block reflector T, 
      __armas_make(&Wrk, C->rows, lb, C->rows, &__armas_data(W)[__armas_size(&T)]);
      __blk_lqmult_right(C, A, tau, &T, &Wrk, flags, lb, conf);
    }
  }
  return 0;
}

/*
 * Calculate required workspace with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 */
int __armas_lqmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  if (flags & ARMAS_RIGHT) {
    return __ws_lqmult_right(A, conf->lb);
  }
  return __ws_lqmult_left(A, conf->lb);
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
