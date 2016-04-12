
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Diagonal matrix

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mult_diag) && defined(__armas_solve_diag)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_scale)  && defined(__armas_invscale)
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

/**
 * @brief Compute \f$ A = A*diag(D) \f$ or \f$ A = diag(D)*A \f$
 *
 * @param A
 *      Target matrix
 * @param D
 *      Diagonal vector or square matrix
 * @param flags
 *      Flag bits, ARMAS_LEFT or ARMAS_RIGHT
 * @param conf
 *      Optional blocking configuration
 */
int __armas_mult_diag(__armas_dense_t *A, __armas_dense_t *D, int flags, armas_conf_t *conf)
{
    __armas_dense_t c, d0, *d;
    int k;

    if (!conf)
        conf = armas_conf_default();
    
    d = D;
    if (! __armas_isvector(D)) {
        __armas_diag(&d0, D, 0);
        d = &d0;
    }
    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_LEFT:
        if (__armas_size(d) != A->rows) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        // scale rows; for each column element-wise multiply of D element
        for (k = 0; k < __armas_size(d); k++) {
            __armas_row(&c, A, k);
            __armas_scale(&c, __armas_get_at(d, k), conf);
        }
        break;
    case ARMAS_RIGHT:
        if (__armas_size(d) != A->cols) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        // scale columns; 
        for (k = 0; k < __armas_size(d); k++) {
            __armas_column(&c, A, k);
            __armas_scale(&c, __armas_get_at(d, k), conf);
        }
        break;
    }
    return 0;
}

/**
 * @brief Compute \f$ A = A*diag(D)^-1 \f$ or \f$ A = diag(D)^-1*A \f$
 *
 * @param A
 *      Target matrix
 * @param D
 *      Diagonal vector or square matrix
 * @param flags
 *      Flag bits, ARMAS_LEFT or ARMAS_RIGHT
 * @param conf
 *      Optional blocking configuration
 */
int __armas_solve_diag(__armas_dense_t *A, __armas_dense_t *D, int flags, armas_conf_t *conf)
{
    __armas_dense_t c, d0, *d;
    int k;

    if (!conf)
        conf = armas_conf_default();
    
    d = D;
    if (! __armas_isvector(D)) {
        __armas_diag(&d0, D, 0);
        d = &d0;
    }
    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_LEFT:
        if (__armas_size(d) != A->rows) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        // scale rows; for each column element-wise multiply of D element
        for (k = 0; k < __armas_size(d); k++) {
            __armas_row(&c, A, k);
            __armas_invscale(&c, __armas_get_at(d, k), conf);
        }
        break;
    case ARMAS_RIGHT:
        if (__armas_size(d) != A->cols) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        // scale columns; 
        for (k = 0; k < __armas_size(d); k++) {
            __armas_column(&c, A, k);
            __armas_invscale(&c, __armas_get_at(d, k), conf);
        }
        break;
    }
    return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

