
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__vec_axpy_ext) && defined(__vec_axpby_ext)
#define __ARMAS_PROVIDES 1
#endif
// extended precision enabled
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"

int __vec_axpy_ext(mvec_t *Y,  const mvec_t *X, DTYPE alpha, int N)
{
    register int i, kx, ky;
    DTYPE y0, y1, x0, x1;
    DTYPE p0, p1, c0, c1;

    for (i = 0; i < N-1; i += 2) {
        twoprod(&x0, &p0, X->md[(i+0)*X->inc], alpha);
        twoprod(&x1, &p1, X->md[(i+1)*X->inc], alpha);

        twosum(&y0, &c0, x0, Y->md[(i+0)*Y->inc]);
        twosum(&y1, &c1, x1, Y->md[(i+1)*Y->inc]);

        Y->md[(i+0)*Y->inc] = (p0 + c0) + y0;
        Y->md[(i+1)*Y->inc] = (p1 + c1) + y1;
    }    
    if (i == N)
        return 0;

    kx = i*X->inc; ky = i*Y->inc;
    twoprod(&x0, &p0, X->md[kx], alpha);
    twosum(&y0, &c0, x0, Y->md[ky]);
    Y->md[ky] = (p0 + c0) + y0;
    return 0;
}

int __vec_axpby_ext(mvec_t *Y,  const mvec_t *X, DTYPE alpha, DTYPE beta, int N)
{
    register int i, kx, ky;
    DTYPE y0, y1, p0, p1, x0, x1, c0, c1;

    if (beta == 1.0)
        return __vec_axpy_ext(Y, X, alpha, N);

    for (i = 0; i < N-1; i += 2) {
        twoprod(&x0, &p0, X->md[(i+0)*X->inc], alpha);
        twoprod(&x1, &p1, X->md[(i+1)*X->inc], alpha);

        twoprod(&y0, &c0, Y->md[(i+0)*Y->inc], beta);
        twoprod(&y1, &c1, Y->md[(i+1)*Y->inc], beta);
        p0 += c0;
        p1 += c1;

        twosum(&y0, &c0, x0, y0);
        twosum(&y1, &c1, x1, y1);

        Y->md[(i+0)*Y->inc] = (p0 + c0) + y0;
        Y->md[(i+1)*Y->inc] = (p1 + c1) + y1;
    }    
    if (i == N)
	return 0;

    kx = i*X->inc; ky = i*Y->inc;
    twoprod(&x0, &p0, X->md[kx], alpha);
    twoprod(&y0, &c0, Y->md[ky], beta);
    p0 += c0;
    twosum(&y0, &c0, x0, y0);
    Y->md[ky] = p0 + c0 + y0;
    return 0;
}

#if 0
/**
 * @brief Compute Y = Y + alpha*X
 *
 * @param[in,out] y target and source vector
 * @param[in]     x source vector
 * @param[in]     alpha scalar multiplier
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup xblas1
 */
int __armas_ex_axpy(__armas_dense_t *y, const __armas_dense_t *x, DTYPE alpha, armas_conf_t *conf)
{
  // only for column or row vectors
  if (!conf)
    conf = armas_conf_default();

  if (x->cols != 1 && x->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (y->cols != 1 && y->rows != 1) {
    conf->error = ARMAS_ENEED_VECTOR;
    return -1;
  }
  if (__armas_size(x) != __armas_size(y)) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
  mvec_t Y       = {y->elems, (y->rows == 1 ? y->step : 1)};

  __vec_axpy_ext(&Y, &X, alpha, __armas_size(y));
  return 0;
}


/**
 * @brief Compute Y = beta*Y + alpha*X
 *
 * @param[in,out] y target and source vector
 * @param[in]     x source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     beta scalar multiplier
 * @param[out]    conf configuration block
 *
 * @retval 0 Ok
 * @retval -1 Failed, conf->error holds error code
 *
 * @ingroup xblas1
 */
int __armas_ex_axpby(__armas_dense_t *y, const __armas_dense_t *x, DTYPE alpha, DTYPE beta, armas_conf_t *conf)
{
    // only for column or row vectors
    if (!conf)
        conf = armas_conf_default();

    if (x->cols != 1 && x->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (y->cols != 1 && y->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (__armas_size(x) != __armas_size(y)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    const mvec_t X = {x->elems, (x->rows == 1 ? x->step : 1)};
    mvec_t Y       = {y->elems, (y->rows == 1 ? y->step : 1)};

    if (beta == 1.0) {
        __vec_axpy_ext(&Y, &X, alpha, __armas_size(y));
    } else {
        __vec_axpby_ext(&Y, &X, alpha, beta, __armas_size(y));
    }
    return 0;
}
#endif

#endif /* __ARMAS_REQUIRES && __ARMAS_PROVIDES */

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

