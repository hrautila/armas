
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>

//#include <armas/armas.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_mult_kernel_inner) && \
    defined(armas_mult_kernel) && defined(armas_mult_kernel_nc)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "matcpy.h"

#include "kernel.h"

/* ---------------------------------------------------------------------------------
 *
 */

void armas_mult_kernel_inner(
    armas_dense_t *Cblk,
    DTYPE alpha,
    const armas_dense_t *Ablk,
    const armas_dense_t *Bblk,
    int rb)
{
    register int j, kp;
    armas_dense_t Ca, Ac;

    for (kp = 0; kp < Cblk->rows; kp += rb) {
        int nK = min(rb, Cblk->rows-kp);
        armas_submatrix_unsafe(&Ca, Cblk, kp, 0, nK, Cblk->cols);
        armas_submatrix_unsafe(&Ac, Ablk, 0, kp, Ablk->rows, nK);
        for (j = 0; j < Cblk->cols-3; j += 4) {
            __CMULT4(&Ca, &Ac, Bblk, alpha, j, nK, Ablk->rows);
        }
        if (j == Cblk->cols)
            continue;
        // the uneven column stripping part ....
        if (j < Cblk->cols-1) {
            __CMULT2(&Ca, &Ac, Bblk, alpha, j, nK, Ablk->rows);
            j += 2;
        }
        if (j < Cblk->cols) {
            __CMULT1(&Ca, &Ac, Bblk, alpha, j, nK, Ablk->rows);
            j++;
        }
    }
}


int armas_mult_kernel(
    DTYPE beta,
    armas_dense_t *C,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *B,
    int flags,
    cache_t *cache)
{
    int ip, jp, kp, nP, nI;
    armas_dense_t Ca, Ac, Bc, Acpy, Bcpy;

    if (alpha == ZERO) {
        if (beta != ONE) {
            armas_scale_unsafe(C, beta);
        }
        return 0;
    }

    int P = (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) != 0 ? A->rows : A->cols;

    // loop over columns of C
    for (jp = 0; jp < C->cols; jp += cache->NB) {
        // in panels of N columns of C, B
        int nJ = min(cache->NB, C->cols-jp);

        // scale C block here.... jp, jp+nJ columns, E-R rows
        armas_submatrix_unsafe(&Ca, C, 0, jp, C->rows, nJ);
        blk_scale(&Ca, beta, C->rows, nJ);

        for (kp = 0; kp < P; kp += cache->KB) {
            nP = min(cache->KB, P-kp);
            armas_make(&Bcpy, nP, nJ, cache->ab_step, cache->Bcpy);
            if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
                armas_submatrix_unsafe(&Bc, B, jp, kp, nJ, nP);
                CPBLK_TRANS(&Bcpy, &Bc, nJ, nP, flags);
            } else {
                armas_submatrix_unsafe(&Bc, B, kp, jp, nP, nJ);
                CPBLK(&Bcpy, &Bc, nP, nJ, flags);
            }

            for (ip = 0; ip < C->rows; ip += cache->MB) {
                nI = min(cache->MB, C->rows-ip);
                armas_make(&Acpy, nP, nI, cache->ab_step, cache->Acpy);
                if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
                    armas_submatrix_unsafe(&Ac, A, kp, ip, nP, nI);
                    CPBLK(&Acpy, &Ac, nP, nI, flags);
                } else {
                    armas_submatrix_unsafe(&Ac, A, ip, kp, nI, nP);
                    CPBLK_TRANS(&Acpy, &Ac, nI, nP, flags);
                }

                armas_submatrix_unsafe(&Ca, C, ip, jp, nI, nJ);
                armas_mult_kernel_inner(&Ca,alpha, &Acpy, &Bcpy, cache->rb);
            }
        }
    }
    return 0;
}

int armas_mult_kernel_nc(
    armas_dense_t *C,
    DTYPE alpha,
    const armas_dense_t *A,
    const armas_dense_t *B,
    int flags,
    cache_t *cache)
{
    if (alpha == ZERO) {
        return 0;
    }

    int ip, jp, kp, nP, nI;
    armas_dense_t Ca, Ac, Bc, Acpy, Bcpy;

    // protect against invalid rb parameter (for time being)
    if (cache->rb == 0 || cache->rb > cache->MB) {
        cache->rb = cache->MB/2;
    }

    int P = (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) != 0 ? A->rows : A->cols;

    for (jp = 0; jp < C->cols; jp += cache->NB) {
        // in panels of N columns of C, B
        int nJ = min(cache->NB, C->cols-jp);

        for (kp = 0; kp < P; kp += cache->KB) {
            nP = min(cache->KB, P-kp);
            armas_make(&Bcpy, nP, nJ, cache->ab_step, cache->Bcpy);
            if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
                armas_submatrix_unsafe(&Bc, B, jp, kp, nJ, nP);
                CPBLK_TRANS(&Bcpy, &Bc, nJ, nP, flags);
            } else {
                armas_submatrix_unsafe(&Bc, B, kp, jp, nP, nJ);
                CPBLK(&Bcpy, &Bc, nP, nJ, flags);
            }

            for (ip = 0; ip < C->rows; ip += cache->MB) {
                nI = min(cache->MB, C->rows-ip);
                armas_make(&Acpy, nP, nI, cache->ab_step, cache->Acpy);
                if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
                    armas_submatrix_unsafe(&Ac, A, kp, ip, nP, nI);
                    CPBLK(&Acpy, &Ac, nP, nI, flags);
                } else {
                    armas_submatrix_unsafe(&Ac, A, ip, kp, nI, nP);
                    CPBLK_TRANS(&Acpy, &Ac, nI, nP, flags);
                }

                armas_submatrix_unsafe(&Ca, C, ip, jp, nI, nJ);
                armas_mult_kernel_inner(&Ca,alpha, &Acpy, &Bcpy, cache->rb);
            }
        }
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
