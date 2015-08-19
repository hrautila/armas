
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_qlmult) 
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
#include "partition.h"


static inline
int __ws_qlmult_left(int M, int N, int lb)
{
  return lb == 0 ? N : lb*(N+lb);
}

static inline
int __ws_qlmult_right(int M, int N, int lb)
{
  return lb == 0 ? M : lb*(M+lb);
}


static int
__unblk_qlmult_left(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                    __armas_dense_t *W, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a01, a11, A22, *Aref;
  __armas_dense_t tT, tB, t0, t1, t2, w12;
  __armas_dense_t CT, CB, C0, c1, C2;
  int pAdir, pAstart, pStart, pDir;
  int mb, nb, tb;

  EMPTY(A00); EMPTY(a11);

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = nb = tb = 0;
    Aref = &ATL;
  } else {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    tb = max(0, __armas_size(tau) - A->cols);
    Aref = &ABR;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_2x1(&CT, 
                  &CB,   /**/  C,   mb, pStart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  __armas_submatrix(&w12, W, 0, 0, C->cols, 1);

  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01,  __nil,
                           __nil, &a11,  __nil, 
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
    __apply_householder2x1(&t1, &a01, &c1, &C0, &w12, ARMAS_LEFT, conf);
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
__blk_qlmult_left(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                  __armas_dense_t *T, __armas_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A01, A11, A22, *Aref, AT;
  __armas_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  __armas_dense_t CT, CB, C0, C1, C2;
  int pAdir, pAstart, pStart, pDir;
  int mb, nb, tb, transpose;

  EMPTY(A00);

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = nb = tb = 0;
    Aref = &ATL;
    transpose = TRUE;
  } else {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    tb = max(0, __armas_size(tau) - A->cols);
    Aref = &ABR;
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
                           &A00,  &A01,  __nil,
                           __nil, &A11,  __nil, 
                           __nil, __nil, &A22,  /**/  A, lb, pAdir);
    __repartition_2x1to3x1(&CT,
                           &C0,
                           &C1,
                           &C2,     /**/ C, A11.cols, pDir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, A11.cols, pDir);
    // ---------------------------------------------------------------------------
    // block reflector for current block
    __merge2x1(&AT, &A01, &A11);
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __armas_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_ql_reflector(&Tcur, &AT, &t1, conf);

    // update with (I - Y*T*Y.T) or (I - Y*T*Y.T).T
    __armas_submatrix(&Wrk, W, 0, 0, C1.cols, A11.cols);
    __update_ql_left(&C1, &C0, &A11, &A01, &Tcur, &Wrk, transpose, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, pAdir);
    __continue_3x1to2x1(&CT,
                        &CB,  /**/  &C0,  &C1,   C,   pDir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}


/*
 * Compute C := C*Q or C := C*Q.T where Q is the M-by-N orthogonal matrix
 * defined K elementary reflectors. (K = A.cols)
 */
static int
__unblk_qlmult_right(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                     __armas_dense_t *W, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, a01, a11, A22, *Aref;
  __armas_dense_t tT, tB, t0, t1, t2, w12;
  __armas_dense_t CL, CR, C0, c1, C2;
  int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
  int mb, nb, tb, cb;

  EMPTY(A00); EMPTY(a11);
  EMPTY(CL); 

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pCstart = ARMAS_PLEFT;
    pCdir   = ARMAS_PRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->cols - A->cols);
    tb = max(0, __armas_size(tau) - min(A->rows, A->cols));
    Aref = &ABR;
  } else {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pCstart = ARMAS_PRIGHT;
    pCdir   = ARMAS_PLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = nb = tb = cb = 0;
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
                           &A00,  &a01,  __nil,
                           __nil, &a11,  __nil, 
                           __nil, __nil, &A22,  /**/  A, 1, pAdir);
    __repartition_1x2to1x3(&CL,
                           &C0,   &c1,   &C2,   /**/ C, 1, pCdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, 1, pDir);
    // ---------------------------------------------------------------------------
    __apply_householder2x1(&t1, &a01, &c1, &C0, &w12, ARMAS_RIGHT, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &a11, &A22,   A, pAdir);
    __continue_1x3to1x2(&CL,   &CR,  /**/  &C0,  &c1,   C,   pCdir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}


static int
__blk_qlmult_right(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                   __armas_dense_t *T, __armas_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABR, A00, A01, A11, A22, *Aref, AT;
  __armas_dense_t tT, tB, t0, t1, t2, w12, Tcur, Wrk;
  __armas_dense_t CL, CR, C0, C1, C2;
  int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
  int mb, nb, tb, cb, transpose;

  EMPTY(A00);
  EMPTY(CL);

  if (flags & ARMAS_TRANS) {
    pAstart = ARMAS_PTOPLEFT;
    pAdir   = ARMAS_PBOTTOMRIGHT;
    pCstart = ARMAS_PLEFT;
    pCdir   = ARMAS_PRIGHT;
    pStart  = ARMAS_PTOP;
    pDir    = ARMAS_PBOTTOM;
    mb = max(0, A->rows - A->cols);
    nb = max(0, A->cols - A->rows);
    cb = max(0, C->cols - A->cols);
    tb = max(0, __armas_size(tau) - min(A->rows, A->cols));
    Aref = &ABR;
    transpose = TRUE;
  } else {
    pAstart = ARMAS_PBOTTOMRIGHT;
    pAdir   = ARMAS_PTOPLEFT;
    pCstart = ARMAS_PRIGHT;
    pCdir   = ARMAS_PLEFT;
    pStart  = ARMAS_PBOTTOM;
    pDir    = ARMAS_PTOP;
    mb = nb = tb = cb = 0;
    Aref = &ATL;
    transpose = FALSE;
  }

  __partition_2x2(&ATL,  __nil,
                  __nil, &ABR,   /**/  A, mb, nb, pAstart);
  __partition_1x2(&CL,   &CR,    /**/  C, mb, pCstart);
  __partition_2x1(&tT, 
                  &tB,   /**/  tau, tb, pStart);
                 
  __armas_submatrix(&w12, W, 0, 0, C->rows, 1);

  while (Aref->rows > 0 && Aref->cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &A01,  __nil,
                           __nil, &A11,  __nil, 
                           __nil, __nil, &A22,  /**/  A, lb, pAdir);
    __repartition_1x2to1x3(&CL,
                           &C0,   &C1,   &C2,   /**/ C, A11.cols, pCdir);
    __repartition_2x1to3x1(&tT,
                           &t0,
                           &t1,
                           &t2,     /**/ tau, lb, pDir);
    // ---------------------------------------------------------------------------
    // build reflector
    __merge2x1(&AT, &A01, &A11);
    __armas_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    __armas_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_ql_reflector(&Tcur, &AT, &t1, conf);

    // update with current block
    __armas_submatrix(&Wrk, W, 0, 0, C1.rows, A11.cols);
    __update_ql_right(&C1, &C0, &A11, &A01, &Tcur, &Wrk, transpose, conf);
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL,  __nil,
                        __nil, &ABR, /**/  &A00, &A11, &A22,   A, pAdir);
    __continue_1x3to1x2(&CL,   &CR,  /**/  &C0,  &C1,   C,   pCdir);
    __continue_3x1to2x1(&tT,
                        &tB,  /**/  &t0,  &t1,   tau, pDir);
  }
  return 0;
}


/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of K first elementary reflectors.
 *
 *    Q = H(k) H(k-1) . . . H(1)
 *
 * as returned by DecomposeQL().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     QL factorization as returned by DecomposeQL() where the upper trapezoidal
 *        part holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors.
 *
 *  W     Workspace matrix, size as returned by WorksizeMultQL().
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block size. If it is zero
 *        unblocked invocation is assumed. Actual blocking size is adjusted
 *        to available workspace size and the smaller of configured block size and
 *        block size implied by workspace is used.
 *
 * Compatible with lapack.DORMQL
 */
int __armas_qlmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                   int flags, armas_conf_t *conf)
{
  WSSIZE wsizer;
  int wsmin, lb, ok, wsneed;
  if (!conf)
    conf = armas_conf_default();

  // default to multiplication from left is nothing defined
  if (!(flags & (ARMAS_LEFT|ARMAS_RIGHT)))
    flags |= ARMAS_LEFT;

  if (flags & ARMAS_RIGHT) {
    ok = C->cols == A->rows;
    wsizer = __ws_qlmult_right;
  } else {
    ok = C->rows == A->rows;
    wsizer = __ws_qlmult_left;
  }

  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  lb = conf->lb;
  wsmin = wsizer(C->rows, C->cols, 0);
  if (! W || __armas_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = wsizer(C->rows, C->cols, lb);
  if (lb > 0 && __armas_size(W) < wsneed) {
    lb = compute_lb(C->rows, C->cols, __armas_size(W), wsizer);
    lb = min(lb, conf->lb);
  }

  if (lb == 0 || A->cols <= lb) {
    // unblocked 
    if (flags & ARMAS_LEFT) {
      __unblk_qlmult_left(C, A, tau, W, flags, conf);
    } else {
      __unblk_qlmult_right(C, A, tau, W, flags, conf);
    }
  } else {
    // blocked code
    __armas_dense_t T, Wrk;

    // space for block reflector
    __armas_make(&T, lb, lb, lb, __armas_data(W));

    if (flags & ARMAS_LEFT) {
      // temporary space after block reflector T, 
      __armas_make(&Wrk, C->cols, lb, C->cols, &__armas_data(W)[__armas_size(&T)]);
      __blk_qlmult_left(C, A, tau, &T, &Wrk, flags, lb, conf);
    } else {
      // temporary space after block reflector T, 
      __armas_make(&Wrk, C->rows, lb, C->rows, &__armas_data(W)[__armas_size(&T)]);
      __blk_qlmult_right(C, A, tau, &T, &Wrk, flags, lb, conf);
    }
  }
  return 0;
}

/*
 * Calculate required workspace with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 */
int __armas_qlmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  if (flags & ARMAS_RIGHT) {
    return __ws_qlmult_right(A->rows, A->cols, conf->lb);
  }
  return __ws_qlmult_left(A->rows, A->cols, conf->lb);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

