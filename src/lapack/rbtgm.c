
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_mult_rbt) && defined(__armas_gen_rbt)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_get_at_unsafe) && defined(__armas_set_at_unsafe)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <stdlib.h>
#include <time.h>

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

#define __RSQRT2 0.7071067811865475

/*
 *   R*A = 1/sqrt(2) * (R0  R1) (A00 A01) = (R0*A00+R1*A10 R0*A01+R1*A11)
 *                     (R0 -R1) (A10 A11)   (R0*A00-R1*A10 R0*A01-R1*A11)
 *
 *   A*R = 1/sqrt(2) * (A00 A01) (R0  R1) = (R0*(A00+A01) R1*(A00-A01))
 *                     (A10 A11) (R0 -R1)   (R0*(A10-A11) R1*(A10-A11))
 *
 */
static
void __rbt_left(__armas_dense_t *A, __armas_dense_t *R, int Nd, int flags, armas_conf_t *conf)
{
    int i, j, rb, nb;
    DTYPE r0, r1, a00, a10;

    // initial 
    if (A->rows == 1) {
        nb = 1; rb = 0;
    } else {
        nb = Nd == 1 ? 1 : (Nd + 1)/2;
        rb = A->rows - nb;
    }

    if (flags & ARMAS_TRANS) {
        //printf("..rows=%d, Nd=%d, nb=%d, rb=%d\n", A->rows, Nd, nb, rb);
        for (j = 0; j < A->cols; j++) {
            for (i = 0; i < rb; i++) {
                r0  = __armas_get_at_unsafe(R, i);
                r1  = __armas_get_at_unsafe(R, i+nb);
                a00 = __armas_get_unsafe(A, i, j);
                a10 = __armas_get_unsafe(A, i+nb, j);
                __armas_set_unsafe(A, i,    j, r0*(a00+a10));
                __armas_set_unsafe(A, i+nb, j, r1*(a00-a10));
            }
            for (; i < nb; i++) {
                r0  = __armas_get_at_unsafe(R, i);
                a00 = __armas_get_unsafe(A, i, j);
                __armas_set_unsafe(A, i,  j, r0*a00);
            }
        }
        return;
    }

    for (j = 0; j < A->cols; j++) {
        for (i = 0; i < rb; i++) {
            r0  = __armas_get_at_unsafe(R, i);
            r1  = __armas_get_at_unsafe(R, i+nb);
            a00 = r0 * __armas_get_unsafe(A, i, j);
            a10 = r1 * __armas_get_unsafe(A, i+nb, j);
            __armas_set_unsafe(A, i,    j, a00+a10);
            __armas_set_unsafe(A, i+nb, j, a00-a10);
        }
        // for case when Nd > rows(A)
        for (; i < nb; i++) {
            r0  = __armas_get_at_unsafe(R, i);
            a00 = __armas_get_unsafe(A, i, j);
            __armas_set_unsafe(A, i, j, r0*a00);
        }
    }
}

/*
 * Update matrix A with partial recursive butterfly matrix R from left.
 */
static
int __rbt_recursive_left(__armas_dense_t *A, __armas_dense_t *R, int flags, armas_conf_t *conf)
{
    __armas_dense_t AT, AB, A0, A1, A2;
    __armas_dense_t RL, RR, R0, R1, R2, rT, rB, r0, r1, r2;
    __armas_dense_t *Rx;
    int pdir, rdir, lb, k;
    int twod = 1 << R->cols;   // twod = 2^d
    int Nd;                    // Nd closest multiple of 2^d larger than N
    
    EMPTY(R0); EMPTY(R1);
    
    // extended size to multiple of 2^d
    Nd = (A->rows + twod - 1) & ~(twod-1);
    k  = twod >> 1;   // initial/final block size is computed with k = twod/2
    if (flags & ARMAS_TRANS) {
        pdir = ARMAS_PLEFT;
        rdir = ARMAS_PRIGHT;
        lb = Nd = Nd / k;
        Rx = &RR;
    } else {
        pdir = ARMAS_PRIGHT;
        rdir = ARMAS_PLEFT;
        lb = A->rows;
        Rx = &RL;
    }
    __partition_1x2(&RL, &RR,  /**/ R, 0, pdir);
    
    while (Rx->cols > 0) {
        __repartition_1x2to1x3(&RL,
                               &R0, &R1, &R2, /**/ R, 1, rdir);
        __partition_2x1(&AT,
                        &AB,  /**/ A, 0, ARMAS_PTOP);
        __partition_2x1(&rT,
                        &rB,  /**/ &R1, 0, ARMAS_PTOP);

        while (AB.rows > 0) {
            __repartition_2x1to3x1(&AT,
                                   &A0,
                                   &A1,
                                   &A2, /**/ A, lb, ARMAS_PBOTTOM);
            __repartition_2x1to3x1(&rT,
                                   &r0,
                                   &r1,
                                   &r2, /**/ &R1, lb, ARMAS_PBOTTOM);
            //----------------------------------------------------------------
            __rbt_left(&A1, &r1, Nd, flags, conf);
            //printf("Nd=%d, lb=%d\n", Nd, lb);
            //---------------------------------------------------------------
            __continue_3x1to2x1(&AT,
                                &AB, /**/ &A0, &A1, /**/ A, ARMAS_PBOTTOM);
            __continue_3x1to2x1(&rT,
                                &rB, /**/ &r0, &r1, /**/ &R1, ARMAS_PBOTTOM);
        }
        __continue_1x3to1x2(&RL, &RR, /**/ &R0, &R1,  /**/ R, rdir);
        //printf("A\n"); __armas_printf(stdout, "%6.3f", A);
        // next butterfly size
        if (flags & ARMAS_TRANS) {
            lb = 2*lb;
            Nd = 2*Nd;
            if (lb > A->rows)
                lb = A->rows;
        } else {
            lb = Nd/2;
            Nd = Nd/2;
        }
    }
    // scale with (1/sqrt(2))^d 
    __armas_mscale(A, __POW(__RSQRT2, R->cols), 0);
    return 0;
}


static
void __rbt_right(__armas_dense_t *A, __armas_dense_t *R, int Nd, int flags, armas_conf_t *conf)
{
    int i, j, rb, nb;
    DTYPE r0, r1, a00, a01;

    // initial 
    if (A->cols == 1) {
        nb = 1; rb = 0;
    } else {
        nb = Nd == 1 ? 1 : (Nd + 1)/2;
        rb = A->cols - nb;
    }

    if (flags & ARMAS_TRANS) {
        j = 0;
        for (i = 0; i < A->rows; i++) {
            for (j = 0; j < rb; j++) {
                r0  = __armas_get_at_unsafe(R, j);
                r1  = __armas_get_at_unsafe(R, j+nb);
                a00 = r0 * __armas_get_unsafe(A, i, j);
                a01 = r1 * __armas_get_unsafe(A, i, j+nb);
                __armas_set_unsafe(A, i, j,    a00+a01);
                __armas_set_unsafe(A, i, j+nb, a00-a01);
            }
        }
        // for case when Nd > rows(A)
        for (; j < nb; j++) {
            r0  = __armas_get_at_unsafe(R, j);
            a00 = __armas_get_unsafe(A, i, j);
            __armas_set_unsafe(A, i, j, r0*a00);
        }
        return;
    }

    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < rb; j++) {
            r0  = __armas_get_at_unsafe(R, j);
            r1  = __armas_get_at_unsafe(R, j+nb);
            a00 = __armas_get_unsafe(A, i, j);
            a01 = __armas_get_unsafe(A, i, j+nb);
            __armas_set_unsafe(A, i, j,    r0*(a00+a01));
            __armas_set_unsafe(A, i, j+nb, r1*(a00-a01));
        }
        for (; j < nb; j++) {
            r0  = __armas_get_at_unsafe(R, j);
            a00 = __armas_get_unsafe(A, i, j);
            __armas_set_unsafe(A, i,  j, r0*a00);
        }
    }
}

/*
 * Update matrix A with partial recursive butterfly matrix R from right.
 *
 * (Note: this version loops over all of A cols(R) times. Small performance speed up
 *  would be to loop over each column/row of A cols(R) times.)
 */
static
int __rbt_recursive_right(__armas_dense_t *A, __armas_dense_t *R, int flags, armas_conf_t *conf)
{
    __armas_dense_t AL, AR, A0, A1, A2;
    __armas_dense_t RL, RR, R0, R1, R2, rT, rB, r0, r1, r2;
    __armas_dense_t *Rx;
    int pdir, rdir, lb, k;
    int twod = 1 << R->cols;   // twod = 2^d
    int Nd;                    // Nd closest multiple of 2^d larger than N

    EMPTY(A0); EMPTY(A1);
    EMPTY(R0); EMPTY(R1);
    
    // extended size to multiple of 2^d
    Nd = (A->cols + twod - 1) & ~(twod-1);
    k  = twod >> 1;   // initial/final block size is computed with k = twod/2
    if (flags & ARMAS_TRANS) {
        pdir = ARMAS_PLEFT;
        rdir = ARMAS_PRIGHT;
        lb = Nd = Nd / k;
        Rx = &RR;
    } else {
        pdir = ARMAS_PRIGHT;
        rdir = ARMAS_PLEFT;
        lb = A->cols;
        Rx = &RL;
    }
    __partition_1x2(&RL, &RR,  /**/ R, 0, pdir);
    
    while (Rx->cols > 0) {
        __repartition_1x2to1x3(&RL,
                               &R0, &R1, &R2, /**/ R, 1, rdir);
        __partition_1x2(&AL, &AR,  /**/ A,   0, ARMAS_PLEFT);
        __partition_2x1(&rT,
                        &rB,       /**/ &R1, 0, ARMAS_PTOP);

        while (AR.cols > 0) {
            __repartition_1x2to1x3(&AL,
                                   &A0, &A1, &A2, /**/ A, lb, ARMAS_PRIGHT);
            __repartition_2x1to3x1(&rT,
                                   &r0,
                                   &r1,
                                   &r2, /**/ &R1, lb, ARMAS_PBOTTOM);
            //----------------------------------------------------------------
            __rbt_right(&A1, &r1, Nd, flags, conf);
            //---------------------------------------------------------------
            __continue_1x3to1x2(&AL, &AR, /**/ &A0, &A1, /**/ A, ARMAS_PRIGHT);
            __continue_3x1to2x1(&rT,
                                &rB,      /**/ &r0, &r1, /**/ &R1, ARMAS_PBOTTOM);
        }
        __continue_1x3to1x2(&RL, &RR, /**/ &R0, &R1,  /**/ R, rdir);
        // next butterfly size
        if (flags & ARMAS_TRANS) {
            lb = 2*lb;
            Nd = 2*Nd;
            if (lb > A->cols)
                lb = A->cols;
        } else {
            lb = Nd/2;
            Nd = Nd/2;
        }
   }
    // scale with (1/sqrt(2))^d 
    __armas_mscale(A, __POW(__RSQRT2, R->cols), 0);
    return 0;
}

/**
 * @brief Multiply matrix A with a partial recursive butterfly matrix R.
 *
 * Computes A = R*A, A = R.T*A, A = A*R or A = A*R.T where R is a partial 
 * recursive butterfly matrix of depth d.
 *
 * Butterfly matrix of size N is defined as matrix
 *
 *   B<n> = 1/sqrt(2) * (R0  R1)  where R0 and R1 is diagonal N/2 matrix.
 *                      (R0 -R1)
 *
 * Partial recursive butterfly matrix U<n,d> of depth d is product of sequence
 *
 *  U<n,d> = B<n/k>*...B<n/2>*B<n>   where k = 2^(d-1).
 *
 * Each element in the sequence is NxN direct sum of k N/k butterfly matrices.
 * Matrix B<n/2> is then NxN direct sum of two N/2 butterfly matrices as below.
 *
 *  B<n/2>  = (B0<n/2>    0   )
 *            (  0     B1<n/2>)
 *
 *  @param[in,out] A
 *     On entry the original matrix. On exit the updated matrix.
 *  @param[in] R
 *     Partial recursive butterfly matrix in banded storage format where
 *     each column of R. Column k of R stores the butterfly matrix at level
 *     N/(R.cols-k)
 *  @param[in] flags
 *     Operator flags, valid bits ARMAS_LEFT, ARMAS_RIGHT, ARMAS_TRANS.
 *  @param[in,out] conf
 *     Configuration block, on error conf.error is set.
 * 
 *  @return 
 *     0 for success, -1 for error.
 */
int __armas_mult_rbt(__armas_dense_t *A, __armas_dense_t *R, int flags, armas_conf_t *conf)
{
    int ok;
    
    if (!conf)
        conf = armas_conf_default();

    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        ok = A->cols == R->rows;
        break;
    case ARMAS_LEFT:
    default:
        ok = A->rows == R->rows;
        break;
    }
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (flags & ARMAS_RIGHT) {
        __rbt_recursive_right(A, R, flags, conf);
    } else {
        __rbt_recursive_left(A, R, flags, conf);
    }
    return 0;
}

/**
 * @brief Compute size Nd such that it is multiple of 2^d.
 *
 *    @param[out] *fact
 *      Pointer to scaling factor that defines initial/final block size of 
 *      recursive butterfly (Nd/fact). If pointer is null factor is not returned.
 *    @param[in] N
 *      Size of matrix
 *    @param[in] depth
 *      Depth of recursive butterfly. 
 */
int __armas_size_rbt(int *fact, int N, int depth)
{
    int Nd;
    int twod = 1 << depth;   // twod = 2^d
    
    // extended size to multiple of 2^d
    Nd = (N + twod - 1) & ~(twod-1);
    // initial/final block size is computed with k = twod/2
    if (*fact)
        *fact  = twod >> 1;   
    return Nd;
}

/*
 * From: Baboulin, Randomization 
 */
void __armas_gen_rbt(__armas_dense_t *R)
{
    static int init = 0;
    double r;
    int i, j;
    
    if (init == 0) {
        srand48((long)time(0));
        init = 1;
    }

    for (j = 0; j < R->cols; j++) {
        for (i = 0; i < R->rows; i++) {
            r = drand48() - 0.5;
            r = exp(r/10.0);
            __armas_set_unsafe(R, i, j, r);
        }
    }
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

