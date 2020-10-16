
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Swap vectors

#include "dtype.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_swap)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"


static
void vec_swap(armas_dense_t * X, armas_dense_t * Y, int N)
{
    register int i, kx, ky;
    register double y0, y1, y2, y3, x0, x1, x2, x3;
    int yinc = Y->rows == 1 ? Y->step : 1;
    int xinc = X->rows == 1 ? X->step : 1;

    for (i = 0; i < N - 3; i += 4) {
        y0 = Y->elems[(i + 0) * yinc];
        y1 = Y->elems[(i + 1) * yinc];
        y2 = Y->elems[(i + 2) * yinc];
        y3 = Y->elems[(i + 3) * yinc];
        x0 = X->elems[(i + 0) * xinc];
        x1 = X->elems[(i + 1) * xinc];
        x2 = X->elems[(i + 2) * xinc];
        x3 = X->elems[(i + 3) * xinc];
        X->elems[(i + 0) * xinc] = y0;
        X->elems[(i + 1) * xinc] = y1;
        X->elems[(i + 2) * xinc] = y2;
        X->elems[(i + 3) * xinc] = y3;
        Y->elems[(i + 0) * yinc] = x0;
        Y->elems[(i + 1) * yinc] = x1;
        Y->elems[(i + 2) * yinc] = x2;
        Y->elems[(i + 3) * yinc] = x3;
    }
    if (i == N)
        return;

    kx = i * xinc;
    ky = i * yinc;
    switch (N - i) {
    case 3:
        y0 = Y->elems[ky];
        Y->elems[ky] = X->elems[kx];
        X->elems[kx] = y0;
        kx += xinc;
        ky += yinc;
    case 2:
        y0 = Y->elems[ky];
        Y->elems[ky] = X->elems[kx];
        X->elems[kx] = y0;
        kx += xinc;
        ky += yinc;
    case 1:
        y0 = Y->elems[ky];
        Y->elems[ky] = X->elems[kx];
        X->elems[kx] = y0;
    }
}

/**
 * @brief Swap vectors X and Y.
 *
 * @param[in,out] X, Y vectors
 * @param[in,out] conf configuration block
 *
 * @retval 0 Ok
 * @retval <0 Failed, conf->error holds error code
 *
 * @ingroup blas
 */
int armas_swap(armas_dense_t * Y, armas_dense_t * X, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    // only for column or row vectors
    if (X->cols != 1 && X->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (Y->cols != 1 && Y->rows != 1) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (armas_size(X) != armas_size(Y)) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    if (armas_size(X) == 0 || armas_size(Y) == 0) {
        return 0;
    }

    vec_swap(Y, X, armas_size(Y));
    return 0;
}
#else
#warning "Missing defines. No code"
#endif /* ARMAS_REQUIRES && ARMAS_PROVIDES */
