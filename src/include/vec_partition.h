
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef ARMAS_VEC_PARTITION_H
#define ARMAS_VEC_PARTITION_H 1


#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE extern inline
#endif

#include "matrix.h"

/*
 * Partition p to 2 by 1 blocks.
 *
 *        xT
 *  x --> --
 *        xB
 *
 * Parameter nb is initial block size for AT (PTOP) or AB (PBOTTOM).  
 */
__ARMAS_INLINE
void vec_partition_2x1(armas_x_dense_t *xT, armas_x_dense_t *xB,
                       const armas_x_dense_t *x, int nb, int side)
{
    int nr = armas_x_size(x);
    if (nb > nr)
        nb = nr;

    switch (side) {
    case ARMAS_PTOP:
        armas_x_subvector_unsafe(xT, x, 0, nb);
        armas_x_subvector_unsafe(xB, x, nb, nr-nb);
        break;
    case ARMAS_PBOTTOM:
    default:
        armas_x_subvector_unsafe(xT, x, 0, nr-nb);
        armas_x_subvector_unsafe(xB, x, nr-nb, nb);
        break;
    }
}

/*
 * Repartition 2 by 1 block to 3 by 1 block.
 *
 *           xT      x0            xT       x0
 * pBOTTOM: --  --> --   ; pTOP:   --  -->  x1
 *           xB      x1            xB       --
 *                   x2                     x2
 *
 */
__ARMAS_INLINE
void vec_repartition_2x1to3x1(armas_x_dense_t *xT, armas_x_dense_t *x0,
                              armas_x_dense_t *x1, armas_x_dense_t *x2,
                              const armas_x_dense_t *x, int nb, int direction)
{
    int nr = armas_x_size(x);
    int nrt = armas_x_size(xT);
    switch (direction) {
    case ARMAS_PBOTTOM:
        if (nb + nrt > nr) {
            nb = nr - nrt;
        }
        armas_x_subvector_unsafe(x0, x, 0, nrt);
        armas_x_subvector_unsafe(x1, x, nrt, nb);
        armas_x_subvector_unsafe(x2, x, nrt+nb, nr-nrt-nb);
        break;
    case ARMAS_PTOP:
    default:
        if (nrt < nb) {
            nb = nrt;
        }
        armas_x_subvector_unsafe(x0, x, 0, nrt-nb);
        armas_x_subvector_unsafe(x1, x, nrt-nb, nb);
        armas_x_subvector_unsafe(x2, x, nrt, nr-nrt);
        break;
    }
}

/*
 * Continue with 2 by 1 block from 3 by 1 block.
 *
 *           xT      x0            xT       x0
 * pBOTTOM: --  <--  x1   ; pTOP:   -- <--  --
 *           xB      --            xB       x1
 *                   x2                     x2
 */
__ARMAS_INLINE
void vec_continue_3x1to2x1(armas_x_dense_t *xT, armas_x_dense_t *xB,
                           armas_x_dense_t *x0, armas_x_dense_t *x1,
                           const armas_x_dense_t *x, int direction)
{
    int nr = armas_x_size(x);
    int nr0 = armas_x_size(x0);
    int nr1 = armas_x_size(x1);
    switch (direction) {
    case ARMAS_PBOTTOM:
        armas_x_subvector_unsafe(xT, x, 0, nr0+nr1);
        armas_x_subvector_unsafe(xB, x, nr0+nr1, nr-nr0-nr1);
        break;
    case ARMAS_PTOP:
    default:
        armas_x_subvector_unsafe(xT, x, 0, nr0);
        armas_x_subvector_unsafe(xB, x, nr0, nr-nr0);
        break;
    }
}
#endif  /* ARMAS_VEC_PARTITION.H */
