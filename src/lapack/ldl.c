
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_ldlfactor)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#include "ldl.h"
/*
 * This file holds the non-pivoting versions of LDL^T factorization.
 */

/*
 *  ( a11  a12 )   ( 1   0   )( d1  0  )( 1  l21.t )
 *  ( a21  A22 )   ( l21 L22 )(  0  D2 )( 0  L22.t )
 *
 *   a11  =   d1
 *   a21  =   l21*d1                       => a21 = a21/d1
 *   A22  =   l21*d1*l21.t + L22*D2*L22.t  => A22 = A22 - l21*d1*l21t
 */
static
int unblk_ldlnp_lower(armas_dense_t * A, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a11, a21, A22;
    int err = 0;
    DTYPE a11val;

    EMPTY(a11);
    EMPTY(ATL);
    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        // d11 = a11; no-op

        // A22 = A22 - l21*d11*l21.T = A22 - a21*a21.T/a11
        a11val = ONE / armas_get_unsafe(&a11, 0, 0);
        armas_mvupdate_trm(ONE, &A22, -a11val, &a21, &a21, ARMAS_LOWER, conf);
        // l21 = a21/a11
        armas_scale(&a21, a11val, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

/*
 *  ( A00  a01 )   ( U00 u01 )( D0  0  )( U00.t 0 )
 *  ( a10  a11 )   (  0   1  )(  0  d1 )( u01.t 1 )
 *
 *   a11  =   d1
 *   a01  =   u01*d1                       => a01 = a01/a11
 *   A00  =   u01*d1*u01.t + U00*D1*U00.t  => A00 = A00 - a01*a01.t/a11
 */
static
int unblk_ldlnp_upper(armas_dense_t * A, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a11, a01, A22;
    int err = 0;
    DTYPE a11val;

    EMPTY(a11);
    EMPTY(ATL);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------
        // d11 = a11; no-op

        // A00 = A00 - u01*d1*u01.T = A00 - a01*a01.T/a11
        a11val = ONE / armas_get_unsafe(&a11, 0, 0);
        armas_mvupdate_trm(ONE, &A00, -a11val, &a01, &a01, ARMAS_UPPER, conf);
        // u01 = a01/a11
        armas_scale(&a01, a11val, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PTOPLEFT);
    }
    return err;
}

/*
 *  ( A11  A12 )   ( L11   0  )( D1  0  )( L11.t  L21.t )
 *  ( A21  A22 )   ( L21  L22 )(  0  D2 )(   0    L22.t )
 *
 *   A11  =   L11*D1*L11.t                 => L11\D1 = LDL(A11)
 *   A12  =   L11*D1*L21.t
 *   A21  =   L21*D1*L11.t                 =>
 *               L21 = A21*(D1*L11.t).-1 = A21*L11.-T*D1.-1
 *   A22  =   L21*D1*L21.t + L22*D2*L22.t  => L22 = A22 - L21*D1*L21.t
 */
static
int blk_ldlnp_lower(armas_dense_t * A, armas_dense_t * W, int lb,
                    armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A11, A21, A22, L21, D;
    int err = 0;

    EMPTY(A00);
    EMPTY(ATL);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, __nil,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        unblk_ldlnp_lower(&A11, conf);
        armas_diag(&D, &A11, 0);

        // A21 = A21*A11.-T
        armas_solve_trm(&A21, ONE, &A11,
                          ARMAS_RIGHT|ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANS, conf);
        // A21 = A21.D.-1  (=L21)
        armas_solve_diag(&A21, ONE, &D, ARMAS_RIGHT, conf);

        // Wrk = L21 = D1*L21.T
        armas_make(&L21, A21.rows, A21.cols, A21.rows, armas_data(W));
        armas_mcopy(&L21, &A21, 0, conf);
        // L21 = L21*D
        armas_mult_diag(&L21, ONE, &D, ARMAS_RIGHT, conf);

        // A22 = A22 - L21*A21.T
        armas_update_trm(ONE, &A22,
                           -ONE, &L21, &A21, ARMAS_LOWER|ARMAS_TRANSB, conf);

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}


static
int blk_ldlnp_upper(armas_dense_t * A, armas_dense_t * W, int lb,
                    armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A01, A11, A22, D, L01;
    int err = 0;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);

    while (ATL.rows > lb) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00,  &A01, __nil,
            __nil, &A11, __nil,
            __nil, __nil, &A22, /**/ A, lb, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------
        unblk_ldlnp_upper(&A11, conf);
        armas_diag(&D, &A11, 0);

        // A01 = A01*A11.-T
        armas_solve_trm(&A01, ONE, &A11,
                          ARMAS_RIGHT|ARMAS_UPPER|ARMAS_UNIT|ARMAS_TRANS, conf);
        // A01 = A01.D.-1  (=L01)
        armas_solve_diag(&A01, ONE, &D, ARMAS_RIGHT, conf);

        // Wrk = L01 = D1*L01.T
        armas_make(&L01, A01.rows, A01.cols, A01.rows, armas_data(W));
        armas_mcopy(&L01, &A01, 0, conf);
        // L01 = L01*D
        armas_mult_diag(&L01, ONE, &D, ARMAS_RIGHT, conf);

        // A00 = A00 - L01*A01.T
        armas_update_trm(ONE, &A00,
                           -ONE, &L01, &A01, ARMAS_UPPER|ARMAS_TRANSB, conf);

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PTOPLEFT);
    }
    if (ATL.rows > 0) {
        unblk_ldlnp_upper(&ATL, conf);
    }
    return err;
}


/*
 * @brief Compute non-pivoting LDLT factoring of symmetric matrix
 *
 * @param A
 *   On entry symmetric matrix store on lower (upper) triangular part. On exit
 *   the LDL.T (UDU.T) factorization of where L (U) is lower (upper) triangular
 *   matrix with unit diagonal and D is stored on diagonal entries.
 * @param W
 *   Working space for blocked implementation. If null or zero sized then
 *   unblocked algorithm used.
 * @param flags
 *   Indicator bits, lower (upper) storage if ARMAS_LOWER (ARMAS_UPPER) set.
 * @param conf
 *   Configuration block
 *
 * @retval  0  ok
 * @retval <0 error
 */
int ldlfactor_np(armas_dense_t * A, armas_dense_t * W, int flags,
                 armas_conf_t * conf)
{
    armas_env_t *env;
    int ws, ws_opt, lb, err = 0;
    if (!conf)
        conf = armas_conf_default();

    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    env = armas_getenv();
    lb = env->lb;
    ws = armas_size(W);
    ws_opt = ws_calculate(A->rows, lb);

    if (ws > 0 && ws < ws_opt) {
        lb = new_lb(A->rows, lb, ws);
    }

    if (lb == 0 || ws == 0 || A->rows <= lb) {
        // unblocked code
        if (flags & ARMAS_UPPER) {
            err = unblk_ldlnp_upper(A, conf);
        } else {
            err = unblk_ldlnp_lower(A, conf);
        }
    } else {
        // blocked version
        if (flags & ARMAS_UPPER) {
            err = blk_ldlnp_upper(A, W, lb, conf);
        } else {
            err = blk_ldlnp_lower(A, W, lb, conf);
        }
    }
    return err;
}

#if defined(ldlsolve_np)
/*
 * @brief Solve X = A*B or X = A.T*B where A is symmetric matrix
 *
 * @param[in,out] B
 *    On entry, input values. On exit, the solutions matrix
 * @param[in] A
 *    The LDL.T (UDU.T) factorized symmetric matrix
 * @param[in] flags
 *    Indicator flags, lower (ARMAS_LOWER) or upper (ARMAS_UPPER) triangular matrix
 * @param[in,out] conf
 *    Configuration block.
 *
 * @retval  0 ok
 * @retval <0 error
 */
int ldlsolve_np(armas_dense_t * B, const armas_dense_t * A, int flags,
                armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (armas_size(B) == 0 || armas_size(A) == 0)
        return 0;

    if (A->rows != A->cols || A->cols != B->rows) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (flags & ARMAS_TRANS) {
        // X = L.-1*(D.-1*(L.-T*B))
        armas_solve_trm(B, ONE, A,
                          flags | ARMAS_UNIT | ARMAS_TRANS | ARMAS_LEFT, conf);
        armas_solve_diag(B, ONE, A, ARMAS_LEFT, conf);
        armas_solve_trm(B, ONE, A, flags | ARMAS_UNIT | ARMAS_LEFT, conf);
    } else {
        // X = L.-T*(D.-1*(L.-1*B))
        armas_solve_trm(B, ONE, A, flags | ARMAS_UNIT | ARMAS_LEFT, conf);
        armas_solve_diag(B, ONE, A, ARMAS_LEFT, conf);
        armas_solve_trm(B, ONE, A, flags | ARMAS_UNIT | ARMAS_TRANS, conf);
    }
    return 0;
}
#endif

#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
