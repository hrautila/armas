
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix norm functions

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_mnorm)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_amax) && defined(armas_asum) && defined(armas_nrm2)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

static
void sum_of_sq(ABSTYPE *ssum, ABSTYPE *scale, armas_dense_t *X, ABSTYPE sum, ABSTYPE scl)
{
    register int i;
    register ABSTYPE a0;

    for (i = 0; i < armas_size(X); i += 1) {
        a0 = armas_get_at_unsafe(X, i);
        if (a0 != ZERO) {
            a0 = ABS(a0);
            if (a0 > scl) {
                sum = ONE + sum * ((scl/a0)*(scl/a0));
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
ABSTYPE matrix_norm_frb(const armas_dense_t *x, armas_conf_t *conf)
{
    int k;
    armas_dense_t v;
    ABSTYPE ssum, scale;

    ssum = ABSONE;
    scale = ABSZERO;
    for (k = 0; k < x->cols; k++) {
        armas_column(&v, x, k);
        sum_of_sq(&ssum, &scale, &v, ssum, scale);
    }
    return scale*SQRT(ssum);
}

/*
 * 1-norm is the maximum of the column sums.
 */
static
ABSTYPE matrix_norm_one(const armas_dense_t *x, armas_conf_t *conf)
{
    int k;
    armas_dense_t v;
    ABSTYPE cmax, amax = ABSZERO;

    for (k = 0; k < x->cols; k++) {
        armas_column(&v, x, k);
        cmax = ABS(armas_asum(&v, conf));
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
ABSTYPE matrix_norm_inf(const armas_dense_t *x, armas_conf_t *conf)
{
    int k;
    armas_dense_t v;
    ABSTYPE cmax, amax = ABSZERO;

    for (k = 0; k < x->rows; k++) {
        armas_row(&v, x, k);
        cmax = ABS(armas_asum(&v, conf));
        if (cmax > amax) {
            amax = cmax;
        }
    }
    return amax;
}


/**
 * @brief Compute norm of general matrix A.
 *
 * @param[in] A
 *    Input matrix
 * @param[in] which
 *    Norm to compute, one of ARMAS_NORM_ONE, ARMAS_NORM_TWO, ARMAS_NORM_INF or
 *    ARMAS_NORM_FRB
 * @param[in] conf
 *    Optional configuration block
 * @return
 *    Value of requested norm.
 * @ingroup matrix
 */
ABSTYPE armas_mnorm(const armas_dense_t *A, int which, armas_conf_t *conf)
{
    ABSTYPE normval = ABSZERO;

    if (!conf)
        conf = armas_conf_default();

    if (! A || armas_size(A) == 0)
        return ABSZERO;

    int is_vector = A->rows == 1 || A->cols == 1;
    switch (which) {
    case ARMAS_NORM_ONE:
        if (is_vector) {
            normval = armas_asum(A, conf);
        } else {
            normval = matrix_norm_one(A, conf);
        }
        break;
    case ARMAS_NORM_TWO:
        if (is_vector) {
            normval = armas_nrm2(A, conf);
        } else {
            conf->error = ARMAS_EIMP;
            normval = ABSZERO;
        }
        break;
    case ARMAS_NORM_INF:
        if (is_vector) {
            normval = armas_amax(A, conf);
        } else {
            normval = matrix_norm_inf(A, conf);
        }
        break;
    case ARMAS_NORM_FRB:
        if (is_vector) {
            normval = armas_nrm2(A, conf);
        } else {
            normval = matrix_norm_frb(A, conf);
        }
        break;
    default:
        conf->error = ARMAS_EINVAL;
        break;
    }
    return normval;
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
