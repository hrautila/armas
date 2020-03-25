
// Copyright (c) Harri Rautila, 2015-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
//#include <armas/armas.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_ext_mult_kernel) && \
    defined(armas_x_ext_mult_kernel_nc)
#define ARMAS_PROVIDES 1
#endif

// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "matcpy.h"

#include "kernel.h"
#include "kernel_ext.h"


// update C block defined by nR rows, nJ columns, nP is A, B common dimension
// A, B data arranged for DOT operations, A matrix is the inner matrix block
// and is looped over nJ times
void armas_x_ext_mult_inner(
    armas_x_dense_t *Cblk,
    armas_x_dense_t *dC,
    DTYPE alpha,
    const armas_x_dense_t *Ablk,
    const armas_x_dense_t *Bblk,
    int rb)
{
    register int j, kp, nK;
    armas_x_dense_t Ca, Da, Ac;

    for (kp = 0; kp < Cblk->rows; kp += rb) {
        nK = min(rb, Cblk->rows-kp);
        armas_x_submatrix_unsafe(&Ca, Cblk, kp, 0, nK, Cblk->cols);
        armas_x_submatrix_unsafe(&Da, dC,   kp, 0, nK, Cblk->cols);
        armas_x_submatrix_unsafe(&Ac, Ablk, 0, kp, Ablk->rows, nK);

        for (j = 0; j < Cblk->cols-1; j += 2) {
            __CMULT2EXT(&Ca, &Da, &Ac, Bblk, alpha, j, nK, Ablk->rows);
        }
        if (j == Cblk->cols)
            continue;

        __CMULT1EXT(&Ca, &Da, &Ac, Bblk, alpha, j, nK, Ablk->rows);
    }
}

// error free update: C + dC += A*B   C is nI,nJ A panel is nI,nP and B panel is nP,nJ
void armas_x_ext_panel_unsafe(
    armas_x_dense_t *C,
    armas_x_dense_t *dC,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    cache_t *cache)
{
    int kp, nK;
    armas_x_dense_t Bc, Ac, Bcpy, Acpy;
    int nJ = C->cols;
    int nI = C->rows;
    int P = flags & (ARMAS_TRANSA|ARMAS_CTRANSA) ? A->rows : A->cols;

    for (kp = 0; kp < P; kp += cache->KB) {
        nK = min(cache->KB, P-kp);

        armas_x_make(&Acpy, nK, nI, cache->ab_step, cache->Acpy);
        armas_x_make(&Bcpy, nK, nJ, cache->ab_step, cache->Bcpy);

        if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
            armas_x_submatrix_unsafe(&Bc, B, 0, kp, nJ, nK);
            CPBLK_TRANS(&Bcpy, &Bc, nJ, nK, flags);
        } else {
            armas_x_submatrix_unsafe(&Bc, B, kp, 0, nK, nJ);
            CPBLK(&Bcpy, &Bc, nK, nJ, flags);
        }

        if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
            armas_x_submatrix_unsafe(&Ac, A, kp, 0, nK, nI);
            CPBLK(&Acpy, &Ac, nK, nI, flags);
        } else {
            armas_x_submatrix_unsafe(&Ac, A, 0, kp, nI, nK);
            CPBLK_TRANS(&Acpy, &Ac, nI, nK, flags);
        }

        armas_x_ext_mult_inner(C, dC, alpha, &Acpy, &Bcpy, cache->rb);
    }
}

// error free update: C + dC += (A + dA)*B   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
// C + dC += A*B; dC += dA*B
void armas_x_ext_panel_dA_unsafe(
    armas_x_dense_t *C,
    armas_x_dense_t *dC,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *dA,
    const armas_x_dense_t *B,
    int flags,
    cache_t *cache)
{
    int kp;
    armas_x_dense_t Acpy, Bcpy, Bc, Ac;
    int nJ = C->cols;
    int nI = C->rows;
    int P = (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) != 0 ? A->rows : A->cols;

    for (kp = 0; kp < P; kp += cache->KB) {
        int nK = min(cache->KB, P-kp);

        armas_x_make(&Acpy, nK, nI, cache->ab_step, cache->Acpy);
        armas_x_make(&Bcpy, nK, nJ, cache->ab_step, cache->Bcpy);

        if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
            armas_x_submatrix_unsafe(&Bc, B, 0, kp, nJ, nK);
            CPBLK_TRANS(&Bcpy, &Bc, nJ, nK, flags);
        } else {
            armas_x_submatrix_unsafe(&Bc, B, kp, 0, nK, nJ);
            CPBLK(&Bcpy, &Bc, nK, nJ, flags);
        }

        if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
            armas_x_submatrix_unsafe(&Ac, A, kp, 0, nK, nI);
            CPBLK(&Acpy, &Ac, nK, nI, flags);
        } else {
            armas_x_submatrix_unsafe(&Ac, A, 0, kp, nI, nK);
            CPBLK_TRANS(&Acpy, &Ac, nI, nK, flags);
        }

        armas_x_ext_mult_inner(C, dC, alpha, &Acpy, &Bcpy, cache->rb);

        // update dC += dA*B
        if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
            armas_x_submatrix_unsafe(&Ac, dA, kp, 0, nK, nI);
            CPBLK(&Acpy, &Ac, nK, nI, flags);
        } else {
            armas_x_submatrix_unsafe(&Ac, dA, 0, kp, nI, nK);
            CPBLK_TRANS(&Acpy, &Ac, nI, nK, flags);
        }

        armas_x_mult_kernel_inner(dC, alpha, &Acpy, &Bcpy, cache->rb);
    }
}


// error free update: C + dC += A*(B + dB)   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
// C + dC += A*B; dC += A*dB
void armas_x_ext_panel_dB_unsafe(
    armas_x_dense_t *C, 
    armas_x_dense_t *dC,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    const armas_x_dense_t *dB,
    int flags,
    cache_t *cache)
{
    int kp;
    armas_x_dense_t Ac, Bc, Acpy, Bcpy;
    int nJ = C->cols;
    int nI = C->rows;
    int P = (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) != 0 ? A->rows : A->cols;

    for (kp = 0; kp < P; kp += cache->KB) {
        int nK = min(cache->KB, P-kp);

        armas_x_make(&Acpy, nK, nI, cache->ab_step, cache->Acpy);
        armas_x_make(&Bcpy, nK, nJ, cache->ab_step, cache->Bcpy);

        if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
            armas_x_submatrix_unsafe(&Bc, B, 0, kp, nJ, nK);
            CPBLK_TRANS(&Bcpy, &Bc, nJ, nK, flags);
        } else {
            armas_x_submatrix_unsafe(&Bc, B, kp, 0, nK, nJ);
            CPBLK(&Bcpy, &Bc, nK, nJ, flags);
        }

        if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
            armas_x_submatrix_unsafe(&Ac, A, kp, 0, nK, nI);
            CPBLK(&Acpy, &Ac, nK, nI, flags);
        } else {
            armas_x_submatrix_unsafe(&Ac, A, 0, kp, nI, nK);
            CPBLK_TRANS(&Acpy, &Ac, nI, nK, flags);
        }

        armas_x_ext_mult_inner(C, dC, alpha, &Acpy, &Bcpy, cache->rb);

        if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
            armas_x_submatrix_unsafe(&Bc, dB, 0, kp, nJ, nK);
            CPBLK_TRANS(&Bcpy, &Bc, nJ, nK, flags);
        } else {
            armas_x_submatrix_unsafe(&Bc, dB, kp, 0, nK, nJ);
            CPBLK(&Bcpy, &Bc, nK, nJ, flags);
        }
        // dC += A*dB
        armas_x_mult_kernel_inner(dC, alpha, &Acpy, &Bcpy, cache->rb);
    }
}


int armas_x_ext_mult_kernel(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    cache_t *cache)
{
    int ip, jp, kp, nP, nI, nJ;
    armas_x_dense_t Ca, Ac, Bc, Acpy, Bcpy, dC;

    int P = (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) != 0 ? A->rows : A->cols;
    // loop over columns of C
    for (jp = 0; jp < C->cols; jp += cache->NB) {
        // in panels of N columns of C, B
        nJ = min(cache->NB, C->cols-jp);

        for (ip = 0; ip < C->rows; ip += cache->MB) {
            nI = min(cache->MB, C->rows-ip);

            armas_x_make(&dC, nI, nJ, cache->ab_step, cache->dC);
            armas_x_submatrix_unsafe(&Ca, C, ip, jp, nI, nJ);
            // compute C0 + dC = beta*C
            if (beta != ONE)
                armas_x_ext_scale_unsafe(&Ca, &dC, beta, &Ca);
            else
                armas_x_scale_unsafe(&dC, ZERO);

            for (kp = 0; kp < P; kp += cache->KB) {
                nP = min(cache->KB, P-kp);

                armas_x_make(&Acpy, nP, nI, cache->ab_step, cache->Acpy);
                armas_x_make(&Bcpy, nP, nJ, cache->ab_step, cache->Bcpy);

                if (flags & (ARMAS_TRANSB|ARMAS_CTRANSB)) {
                    armas_x_submatrix_unsafe(&Bc, B, jp, kp, nJ, nP);
                    CPBLK_TRANS(&Bcpy, &Bc, nJ, nP, flags);
                } else {
                    armas_x_submatrix_unsafe(&Bc, B, kp, jp, nP, nJ);
                    CPBLK(&Bcpy, &Bc, nP, nJ, flags);
                }

                if (flags & (ARMAS_TRANSA|ARMAS_CTRANSA)) {
                    armas_x_submatrix_unsafe(&Ac, A, kp, ip, nP, nI);
                    CPBLK(&Acpy, &Ac, nP, nI, flags);
                } else {
                    armas_x_submatrix_unsafe(&Ac, A, ip, kp, nI, nP);
                    CPBLK_TRANS(&Acpy, &Ac, nI, nP, flags);
                }

                armas_x_ext_mult_inner(&Ca, &dC, alpha, &Acpy, &Bcpy, cache->rb);
            }
            // merge back to target
            armas_x_merge2_unsafe(&Ca, &Ca, &dC);
        }
    }
    return 0;
}

int armas_x_ext_mult_kernel_nc(
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    cache_t *cache)
{
    return armas_x_ext_mult_kernel(ONE, C, alpha, A, B, flags, cache);
}

/*
 * We need 4 intermediate areas: C0, dC where C = C0 + dC, A0 holding current block in A,
 *
 * strategy: [t,p] tiles of C compute for each tile
 *   C = C0 + dC = sum alpha*A[i,k]*B[k,j]
 *      where A[i,:] is row panel of A 
 *      and   B[:,j] is column panel of B
 */

#else
#warning "Missing defines; no code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
