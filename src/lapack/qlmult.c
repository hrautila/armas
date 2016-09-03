
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Multiply with Q matrix 

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qlmult) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__householder) 
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
__unblk_qlmult_left(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                    armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a01, a11, A22, *Aref;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;
  armas_x_dense_t CT, CB, C0, c1, C2;
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
    tb = max(0, armas_x_size(tau) - A->cols);
    Aref = &ABR;
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
__blk_qlmult_left(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                  armas_x_dense_t *T, armas_x_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A01, A11, A22, *Aref, AT;
  armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
  armas_x_dense_t CT, CB, C0, C1, C2;
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
    tb = max(0, armas_x_size(tau) - A->cols);
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
    armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    armas_x_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_ql_reflector(&Tcur, &AT, &t1, conf);

    // update with (I - Y*T*Y.T) or (I - Y*T*Y.T).T
    armas_x_submatrix(&Wrk, W, 0, 0, C1.cols, A11.cols);
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
__unblk_qlmult_right(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                     armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, a01, a11, A22, *Aref;
  armas_x_dense_t tT, tB, t0, t1, t2, w12;
  armas_x_dense_t CL, CR, C0, c1, C2;
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
    tb = max(0, armas_x_size(tau) - min(A->rows, A->cols));
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
                 
  armas_x_submatrix(&w12, W, 0, 0, C->rows, 1);

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
__blk_qlmult_right(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                   armas_x_dense_t *T, armas_x_dense_t *W, int flags, int lb, armas_conf_t *conf)
{
  armas_x_dense_t ATL, ABR, A00, A01, A11, A22, *Aref, AT;
  armas_x_dense_t tT, tB, t0, t1, t2, w12, Tcur, Wrk;
  armas_x_dense_t CL, CR, C0, C1, C2;
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
    tb = max(0, armas_x_size(tau) - min(A->rows, A->cols));
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
                 
  armas_x_submatrix(&w12, W, 0, 0, C->rows, 1);

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
    armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
    armas_x_mscale(&Tcur, 0.0, ARMAS_ANY);
    __unblk_ql_reflector(&Tcur, &AT, &t1, conf);

    // update with current block
    armas_x_submatrix(&Wrk, W, 0, 0, C1.rows, A11.cols);
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


/**
 * \brief Multiply with orthogonal Q matrix 
 *
 * Multiply and replace C with \f$ QC \f$ or \f$ Q^TC \f$ where Q is a real orthogonal matrix
 * defined as the product of K first elementary reflectors.
 *
 *    \f$ Q = H_k H_{k-1} ... H_k \f$
 *
 * as returned by qlfactor().
 *
 * \param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then N-by-M matrix
 *     On exit C is overwritten by \f$ QC \f$ or \f$ Q^TC \f$. If bit *ARMAS_LEFT* is 
 *     set then C is overwritten by \f$ CQ \f$ or \f$ CQ^T \f$
 *
 * \param[in] A
 *     QL factorization as returned by qrfactor() where the upper trapezoidal
 *     part holds the elementary reflectors.
 *
 * \param[in] tau
 *    The scalar factors of the elementary reflectors.
 *
 * \param[out] W
 *     Workspace matrix, size as returned by qrmult_work().
 *
 * \param[in] flags
 *    Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT* and *ARMAS_TRANS*
 *       
 * \param[in] conf
 *   Blocking configuration. Field LB defines block size. If it is zero
 *   unblocked invocation is assumed. Actual blocking size is adjusted
 *   to available workspace size and the smaller of configured block size and
 *   block size implied by workspace is used.
 *
 * \retval  0 Succes
 * \retval -1 Failure, `conf.error` holds error code.
 *
 * Compatible with lapack.DORMQL
 */
int armas_x_qlmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
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
  if (! W || armas_x_size(W) < wsmin) {
    conf->error = ARMAS_EWORK;
    return -1;
  }
  // adjust blocking factor for workspace
  wsneed = wsizer(C->rows, C->cols, lb);
  if (lb > 0 && armas_x_size(W) < wsneed) {
    lb = compute_lb(C->rows, C->cols, armas_x_size(W), wsizer);
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
    armas_x_dense_t T, Wrk;

    // space for block reflector
    armas_x_make(&T, lb, lb, lb, armas_x_data(W));

    if (flags & ARMAS_LEFT) {
      // temporary space after block reflector T, 
      armas_x_make(&Wrk, C->cols, lb, C->cols, &armas_x_data(W)[armas_x_size(&T)]);
      __blk_qlmult_left(C, A, tau, &T, &Wrk, flags, lb, conf);
    } else {
      // temporary space after block reflector T, 
      armas_x_make(&Wrk, C->rows, lb, C->rows, &armas_x_data(W)[armas_x_size(&T)]);
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
int armas_x_qlmult_work(armas_x_dense_t *A, int flags, armas_conf_t *conf)
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

