
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix norm functions

//! \cond
#include "dtype.h"
//! \endcond

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mnorm) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_amax) && defined(__armas_asum) && defined(__armas_nrm2)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
//! \endcond

static 
void __sum_of_sq(ABSTYPE *ssum, ABSTYPE *scale, __armas_dense_t *X, ABSTYPE sum, ABSTYPE scl)
{
  register int i;
  register ABSTYPE a0;

  for (i = 0; i < __armas_size(X); i += 1) {
    a0 = __armas_get_at_unsafe(X, i);
    if (a0 != __ZERO) {
      a0 = __ABS(a0);
      if (a0 > scl) {
        sum = __ONE + sum * ((scl/a0)*(scl/a0));
        scl = a0;
      } else {
        sum = sum + (a0/scl)*(a0/scl);
      }
    }
  }    
  *ssum = sum;
  *scale = scl;
}

/*
 * Frobenius-norm is the square root of sum of the squares of the matrix elements.
 */
static
ABSTYPE __matrix_norm_frb(const __armas_dense_t *x, armas_conf_t *conf)
{
  int k;
  __armas_dense_t v;
  ABSTYPE ssum, scale;

  ssum = __ABSONE;
  scale = __ABSZERO;
  for (k = 0; k < x->cols; k++) {
    __armas_column(&v, x, k);
    __sum_of_sq(&ssum, &scale, &v, ssum, scale);
  }
  return scale*__SQRT(ssum);
}

/*
 * 1-norm is the maximum of the column sums.
 */
static
ABSTYPE __matrix_norm_one(const __armas_dense_t *x, armas_conf_t *conf)
{
  int k;
  __armas_dense_t v;
  ABSTYPE cmax, amax = __ABSZERO;

  for (k = 0; k < x->cols; k++) {
    __armas_column(&v, x, k);
    cmax = __ABS(__armas_asum(&v, conf));
    if (cmax > amax) {
      amax = cmax;
    }
  }
  return amax;
}

/*
 * inf-norm is the maximum of the row sums.
 */
static
ABSTYPE __matrix_norm_inf(const __armas_dense_t *x, armas_conf_t *conf)
{
  int k;
  __armas_dense_t v;
  ABSTYPE cmax, amax = __ABSZERO;

  for (k = 0; k < x->rows; k++) {
    __armas_row(&v, x, k);
    cmax = __ABS(__armas_asum(&v, conf));
    if (cmax > amax) {
      amax = cmax;
    }
  }
  return amax;
}


/**
 * \brief Compute norm of general matrix A.
 *
 * \param[in] A 
 *    Input matrix
 * \param[in] which 
 *    Norm to compute, one of ARMAS_NORM_ONE, ARMAS_NORM_TWO, ARMAS_NORM_INF or
 *    ARMAS_NORM_FRB
 * \param[in] conf 
 *    Optional configuration block
 * \ingroup matrix
 */
ABSTYPE __armas_mnorm(const __armas_dense_t *A, int which, armas_conf_t *conf)
{
  ABSTYPE normval = __ABSZERO;

  if (!conf)
    conf = armas_conf_default();

  if (! A || __armas_size(A) == 0)
    return __ABSZERO;

  int is_vector = A->rows == 1 || A->cols == 1;
  switch (which) {
  case ARMAS_NORM_ONE:
    if (is_vector) {
      normval = __armas_asum(A, conf);
    } else {
      normval = __matrix_norm_one(A, conf);
    }
    break;
  case ARMAS_NORM_TWO:
    if (is_vector) {
      normval = __armas_nrm2(A, conf);
    } else {
      conf->error = ARMAS_EIMP;
      normval = __ABSZERO;
    }
    break;
  case ARMAS_NORM_INF:
    if (is_vector) {
      normval = __ABS(__armas_amax(A, conf));
    } else {
      normval = __matrix_norm_inf(A, conf);
    }
    break;
  case ARMAS_NORM_FRB:
    if (is_vector) {
      normval = __armas_nrm2(A, conf);
    } else {
      normval = __matrix_norm_frb(A, conf);
    }
    break;
  default:
    conf->error = ARMAS_EINVAL;
    break;
  }
  return normval;
}


static
ABSTYPE __trm_norm_one(const __armas_dense_t *A, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ABR, ATR, A00, a01, a11, a21, A22, *Acol;
  ABSTYPE aval, cmax = __ABSZERO;

  Acol = flags & ARMAS_UPPER ? &a01 : &a21;

  EMPTY(A00); EMPTY(a11);

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  &a01, __nil,
                           __nil, &a11, __nil,
                           __nil, &a21, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------

    aval = flags & ARMAS_UNIT
      ? __ABSONE
      : __ABS(__armas_get_at_unsafe(&a11, 0));
    aval += __armas_asum(Acol, conf);
    if (aval > cmax)
      cmax = aval;
    
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }
  return cmax;
}

static
ABSTYPE __trm_norm_inf(const __armas_dense_t *A, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ABR, ATR, A00, a10, a11, a12, A22, *Arow;
  ABSTYPE aval, cmax = __ABSZERO;

  Arow = flags & ARMAS_UPPER ? &a12 : &a10;

  EMPTY(A00); EMPTY(a11);

  __partition_2x2(&ATL, &ATR,
                  &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

  while (ABR.rows > 0 && ABR.cols > 0) {
    __repartition_2x2to3x3(&ATL,
                           &A00,  __nil, __nil,
                           &a10,  &a11,  &a12,
                           __nil, __nil, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
    // ---------------------------------------------------------------------------

    aval = flags & ARMAS_UNIT
      ? __ABSONE
      : __ABS(__armas_get_at_unsafe(&a11, 0));
    aval += __armas_asum(Arow, conf);
    if (aval > cmax)
      cmax = aval;
    
    // ---------------------------------------------------------------------------
    __continue_3x3to2x2(&ATL, &ATR,
                        &ABL, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
  }
  return cmax;
}

static
ABSTYPE __trm_norm_frb(const __armas_dense_t *A, int flags, armas_conf_t *conf)
{
  __armas_dense_t ATL, ABL, ABR, ATR, A00, a01, a11, a21, A22, *Acol;
    ABSTYPE ssum, scale;

    EMPTY(A00); EMPTY(a11);
    
    ssum = __ABSONE;
    scale = __ABSZERO;

    Acol = flags & ARMAS_UPPER ? &a01 : &a21;

    __partition_2x2(&ATL, &ATR,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00,  &a01, __nil,
                               __nil, &a11, __nil,
                               __nil, &a21, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------------
        if (flags & ARMAS_UNIT) {
          ssum += __ABSONE;
        } else {
          __sum_of_sq(&ssum, &scale, &a11, ssum, scale);
        }
        __sum_of_sq(&ssum, &scale, Acol, ssum, scale);
        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, &ATR,
                            &ABL, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    }
    return scale*__SQRT(ssum);
}

/**
 * \brief Compute norm of general or triangular matrix A.
 *
 * \param[in] A 
 *    Input matrix
 * \param[in] which 
 *    Norm to compute, one of ARMAS_NORM_ONE, ARMAS_NORM_TWO, ARMAS_NORM_INF or
 *    ARMAS_NORM_FRB. 
 * \param[in] flags
 *    Matrix type indicator, ARMAS_LOWER, ARMAS_UPPER or zero.
 * \param[in] conf 
 *    Optional configuration block
 *
 * \returns Computed norm. 
 *
 * (Note ARMAS_NORM_TWO not implemented.)
 * \ingroup matrix
 */
ABSTYPE __armas_norm(const __armas_dense_t *A, int which, int flags, armas_conf_t *conf)
{
  DTYPE normval = __ABSZERO;

  if (!conf)
    conf = armas_conf_default();

  if (! A || __armas_size(A) == 0)
    return __ABSZERO;

  if (!(flags & (ARMAS_LOWER|ARMAS_UPPER))) {
    return __armas_mnorm(A, which, conf);
  }
  // triangular/trapezoidial matrices here
  switch (which) {
  case ARMAS_NORM_ONE:
    normval = __trm_norm_one(A, flags, conf);
    break;
  case ARMAS_NORM_TWO:
    conf->error = ARMAS_EIMP;
    normval = __ABSZERO;
    break;
  case ARMAS_NORM_INF:
    normval = __trm_norm_inf(A, flags, conf);
    break;
  case ARMAS_NORM_FRB:
    normval = __trm_norm_frb(A, flags, conf);
    break;
  default:
    conf->error = ARMAS_EINVAL;
    break;
  }
  return normval;
  
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:

