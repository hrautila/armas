
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_update2_rbt)  && defined(__update2_rbt_descend)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mscale) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

/*
 * Compute R.T*A*S
 *
 *    1/2* (R0  R0) (A00 A01) (S0  S1)   =>
 *         (R1 -R1) (A10 A11) (S0 -S1)
 *
 *    1/2* (R0*(A00+A10) R0*(A01+A11)) (S0  S1)  =>
 *         (R1*(A00-A10) R1*(A01-A11)) (S0 -S1)
 *
 *    1/2* (R0*(A00+A01+A10+A11)*S0  R0*(A00+A10-A01-A11)*S1)  
 *         (R1*(A00-A10+A01-A11)*S0  R1*(A00-A10-A01+A11)*S1)
 *
 *    1/2* (R0*(A00+A01+A10+A11)*S0  R0*(A00-A01+A10-A11)*S1)  
 *         (R1*(A00+A01-A10-A11)*S0  R1*(A00-A01-A10+A11)*S1)
 *
 *  Update A as if it were partition follow
 *
 *            ATL | ATR
 *     A =   -----+----  ; ATL [lb,lb], ATR [lb,nb], ABL [mb,lb] and ABR [mb,nb]
 *            ABL | ABR 
 *
 *     lb >= mb == nb
 */
static
void __update2_rbt(__armas_dense_t *A, __armas_dense_t *R,
                  __armas_dense_t *S, int Nd, armas_conf_t *conf)
{
    int lb, nb, mb, i, j;
    DTYPE r0, r1, s0, s1;
    DTYPE a00, a01, a10, a11;

    // A00 = [lb.lb] A01 = [lb,A.cols-lb], A10 = [A.rows-lb,lb], A11=[A.rows-lb,A.cols-lb]
    // lb >= mb && lb >= nb && mb == nb;
    lb = Nd/2;
    mb = A->rows - lb;
    nb = A->cols - lb;
    for (j = 0; j < nb; j++) {
        s0  = __armas_get_at_unsafe(S, j);
        s1  = __armas_get_at_unsafe(S, j+lb);
        for (i = 0; i < mb; i++) {
            r0  = __armas_get_at_unsafe(R, i);
            r1  = __armas_get_at_unsafe(R, i+lb);
            a00 = __armas_get_unsafe(A, i,    j);
            a01 = __armas_get_unsafe(A, i,    j+lb);
            a10 = __armas_get_unsafe(A, i+lb, j);
            a11 = __armas_get_unsafe(A, i+lb, j+lb);
            __armas_set_unsafe(A, i,    j,    s0*r0*((a00+a01)+(a10+a11)));
            __armas_set_unsafe(A, i,    j+lb, s1*r0*((a00+a10)-(a01+a11))); // a00-a01+a10-a11
            __armas_set_unsafe(A, i+lb, j,    s0*r1*((a00+a01)-(a10+a11))); // a00+a01-a10-a11
            __armas_set_unsafe(A, i+lb, j+lb, s1*r1*((a00+a11)-(a01+a10))); // a00-a01-a10+a11
         }
        // for case when Nd > rows(A); here i+lb > rows(A)
        for (; i < lb; i++) {
            r0  = __armas_get_at_unsafe(R, i);
            a00 = __armas_get_unsafe(A, i,    j);
            a01 = __armas_get_unsafe(A, i,    j+lb);
            __armas_set_unsafe(A, i,    j,    s0*r0*(a00+a01));
            // s1 == 0; s1*r0 == 0
            //__armas_set_unsafe(A, i,    j+lb, s1*r0*(a00-a01+a10-a11));
        }
    }
    // Nd > cols(A); here j+lb > cols(A); this updates first and second quadrant
    for (; j < lb; j++) {
        s0  = __armas_get_at_unsafe(S, j);
        //s1  = __armas_get_at_unsafe(S, j+lb);
        for (i = 0; i < mb; i++) {
            r0  = __armas_get_at_unsafe(R, i);
            r1  = __armas_get_at_unsafe(R, i+lb);
            a00 = __armas_get_unsafe(A, i,    j);
            a10 = __armas_get_unsafe(A, i+lb, j);
            __armas_set_unsafe(A, i,    j,    s0*r0*(a00+a10));
            __armas_set_unsafe(A, i+lb, j,    s0*r1*(a00-a10));
        }
        // for case when Nd > rows(A)
        for (; i < lb; i++) {
            r0  = __armas_get_at_unsafe(R, i);
            a00 = __armas_get_unsafe(A, i,    j);
            __armas_set_unsafe(A, i,    j,    s0*r0*(a00));
        }

    }

}

int __update2_rbt_descend(__armas_dense_t *A, __armas_dense_t *U, 
                          __armas_dense_t *V, int Nd, armas_conf_t *conf)
{
    __armas_dense_t A00, A01, A10, A11;
    __armas_dense_t UL, UR, U0, U1, VL, VR, V0, V1;
    int lb;

    EMPTY(VL); EMPTY(VR);
    EMPTY(UL); EMPTY(UR);
    
    //printf("..update_descend: Nd=%d\n", Nd);
    
    if (U->cols == 1 || V->cols == 1) {
        // end of recursion; update current block
        __update2_rbt(A, U, V, Nd, conf);
        return 0;
    }

    lb = Nd/2;
    
    // start from highest level
    __partition_1x2(&UL,  &UR,  /**/ U, 1, ARMAS_PRIGHT);
    __partition_1x2(&VL,  &VR,  /**/ V, 1, ARMAS_PRIGHT);

    __partition_2x1(&U0,
                    &U1,  /**/ &UL, lb, ARMAS_PTOP);
    __partition_2x1(&V0,
                    &V1,  /**/ &VL, lb, ARMAS_PTOP);

    __partition_2x2(&A00, &A01,
                    &A10, &A11, /**/ A, lb, lb, ARMAS_PTOP);

    // compute subblocks;
    __update2_rbt_descend(&A00, &U0, &V0, Nd/2, conf);
    __update2_rbt_descend(&A01, &U0, &V1, Nd/2, conf);
    __update2_rbt_descend(&A10, &U1, &V0, Nd/2, conf);
    __update2_rbt_descend(&A11, &U1, &V1, Nd/2, conf);
    
    // update current level
    __update2_rbt(A, &UR, &VR, Nd, conf);
    return 0;
}

/**
 * @brief Compute A = U.T*A*V where U and V are partial recursive butterfly matrices.
 *
 * @param[in,out] A
 *   On entry the initial matrix. On exit the updated matrix.
 * @param[in] U
 *   Partial recursive butterfly matrix
 * @param[in] V
 *   Partial recursive butterfly matrix
 * @param[in,out] conf
 *   Configuration block.
 *
 *   U<n> * (U0<n/2>   0   ) (A00 A01)  (V0<n/2>    0   ) * V<n>
 *          (   0   U1<n/2>) (A10 A11)  (   0    V1<n/2>)
 * 
 *   U<n> * (U0<n/2>*A00*V0<n/2>  U0<n/2>*A01*V1<n/2>) * V<n>
 *          (U1<n/2>*A10*V0<n/2>  U1<n/2>*A11*V1<n/2>)
 * 
 */
int __armas_update2_rbt(__armas_dense_t *A, 
                        __armas_dense_t *U, __armas_dense_t *V, armas_conf_t *conf)
{
    int ok, Nd, twod;
    
    if (!conf)
        conf = armas_conf_default();

    ok = A->cols == A->rows && U->rows == V->rows 
        && U->cols == V->cols && U->rows == A->rows;
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    
    // extended size to multiple of 2^d
    twod = 1 << U->cols;
    Nd = (A->cols + twod - 1) & ~(twod-1);
    
    __update2_rbt_descend(A, U, V, Nd, conf);
    __armas_mscale(A, __POW(0.5, U->cols), 0);
    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

