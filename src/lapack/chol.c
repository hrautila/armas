
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Cholesky factorization

#include "dtype.h"
#include "dlpack.h"

// ---------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_cholfactor) && defined(armas_x_cholfactor_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas1)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"
//! \endcond

static
int unblk_cholfactor_lower(armas_x_dense_t * A, armas_conf_t * cf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a21, A22;
    int err = 0;
    DTYPE a11val;

    EMPTY(A00);
    EMPTY(a11);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        // --------------------------------------------------------------------
        // a21 = a21/a11
        a11val = armas_x_get(&a11, 0, 0);
        if (a11val >= 0.0) {
            // a11 = sqrt(a11)
            a11val = SQRT(a11val);
            armas_x_set(&a11, 0, 0, a11val);

            // a21 = a21/a11
            armas_x_scale(&a21, ONE/a11val, cf);

            // A22 = A22 - a21*a21.T
            armas_x_mvupdate_sym(ONE, &A22, -ONE, &a21, ARMAS_LOWER, cf);
        } else {
            if (err == 0) {
                cf->error =
                    a11val < 0.0 ? ARMAS_ENEGATIVE : ARMAS_ESINGULAR;
                err = -1;
            }
        }

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

static
int blk_cholfactor_lower(armas_x_dense_t * A, int lb,
                           armas_conf_t * cf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A21, A22;
    int err = 0;

    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, __nil,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        // --------------------------------------------------------------------
        // A11 = CHOL(A11)
        if (unblk_cholfactor_lower(&A11, cf) != 0) {
            err = err == 0 ? -1 : err;
        }
        // A21 = A21 * tril(A11).-T
        armas_x_solve_trm(
            &A21, ONE, &A11, ARMAS_RIGHT | ARMAS_LOWER | ARMAS_TRANSA, cf);

        // A22 = A22 - A21*A21.T
        armas_x_update_sym(ONE, &A22, -ONE, &A21, ARMAS_LOWER, cf);

        // --------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }

    // last block with unblocked
    if (ABR.rows > 0) {
        if (unblk_cholfactor_lower(&ABR, cf) != 0) {
            err = err == 0 ? -1 : err;
        }
    }
    return err;
}


static
int unblk_cholfactor_upper(armas_x_dense_t * A, armas_conf_t * cf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, A22;
    int err = 0;
    DTYPE a11val;

    EMPTY(A00);
    EMPTY(a11);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        // a21 = a21/a11
        a11val = armas_x_get(&a11, 0, 0);
        if (a11val > 0.0) {
            // a11 = sqrt(a11)
            a11val = SQRT(a11val);
            armas_x_set(&a11, 0, 0, a11val);

            // a12 = a12/a11
            armas_x_scale(&a12, ONE/a11val, cf);

            // A22 = A22 - a12*a12.T
            armas_x_mvupdate_sym(ONE, &A22, -ONE, &a12, ARMAS_UPPER, cf);
        } else {
            if (err == 0) {
                cf->error =
                    a11val < 0.0 ? ARMAS_ENEGATIVE : ARMAS_ESINGULAR;
                err = -1;
            }
        }

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

static
int blk_cholfactor_upper(armas_x_dense_t * A, int lb, armas_conf_t * cf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A22;
    int err = 0;

    EMPTY(A00);
    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, __nil, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        // A11 = CHOL(A11)
        if (unblk_cholfactor_upper(&A11, cf) != 0) {
            err = err == 0 ? -1 : err;
        }
        // A12 = tril(A11).-T * A12
        armas_x_solve_trm(
            &A12, ONE, &A11, ARMAS_LEFT | ARMAS_UPPER | ARMAS_TRANSA, cf);

        // A22 = A22 - A12.T*A12
        armas_x_update_sym(
            ONE, &A22, -ONE, &A12, ARMAS_UPPER | ARMAS_TRANSA, cf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A,  ARMAS_PBOTTOMRIGHT);
    }

    if (ABR.rows > 0) {
        if (unblk_cholfactor_upper(&ABR, cf) != 0) {
            err = err == 0 ? -1 : err;
        }
    }
    return err;
}

extern
int armas_x_cholfactor_pv(armas_x_dense_t * A, armas_x_dense_t * W,
                    armas_pivot_t * P, int flags, armas_conf_t * cf);

extern
int armas_x_cholsolve_pv(armas_x_dense_t * B, armas_x_dense_t * A,
                   armas_pivot_t * P, int flags, armas_conf_t * cf);

/**
 * @brief Cholesky factorization
 *
 * @see armas_x_cholfactor_w
 * @ingroup lapack
 */
int armas_x_cholfactor(armas_x_dense_t * A,
                       armas_pivot_t * P, int flags, armas_conf_t * cf)
{
    int err = 0;
    armas_wbuf_t wb = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (P != ARMAS_NOPIVOT) {
        if (!armas_walloc(&wb, A->cols * sizeof(DTYPE))) {
            cf->error = ARMAS_EMEMORY;
            return -ARMAS_EMEMORY;
        }
        err = armas_x_cholfactor_w(A, P, flags, &wb, cf);
        armas_wrelease(&wb);
        return err;
    }

    err = armas_x_cholfactor_w(A, ARMAS_NOPIVOT, flags, ARMAS_NOWORK, cf);
    return err;
}

/**
 * @brief Non pivoting Cholesky factorization
 *
 * Compute the Cholesky factorization of a symmetric positive definite
 * N-by-N matrix A.
 *
 * @param[in, out] A
 *   On entry symmetric posivitive definite matrix. On exit Cholesky
 *   factorization of matrix.
 * @param[in] flags
 *   Matrix structure indicator *ARMAS_LOWER* or *ARMAS_UPPER*
 * @param[in,out] cf
 *   Configuration block
 * @retval  0 Success
 * @retval <0 Failure
 *
 * @ingroup lapack
 */
int armas_x_cholesky(armas_x_dense_t * A, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    return armas_x_cholfactor_w(A, ARMAS_NOPIVOT, flags, ARMAS_NOWORK, cf);
}

/**
 * @brief Cholesky factorization
 *
 * Compute the Cholesky factorization of a symmetric positive definite
 *  N-by-N matrix A.
 *
 * @param[in,out] A
 *      On entry, the symmetric matrix A. If *ARMAS_UPPER* is set
 *      the upper triangular part of A contains the upper triangular
 *      part of the matrix A, and strictly lower part A is not referenced
 *      If *ARMAS_LOWER* is set the lower triangular part of a contains the
 *      lower triangular part of the matrix A. Likewise, the strictly upper
 *      part of A is not referenced. On exit, factor U or L from the
 *      Cholesky factorization \f$ A = U^T U \f$ or \f$ A = L L^T \f$
 * @param[out] P
 *      Optional pivot array. If non null then pivoting factorization is
 *      computed. Set to ARMAS_NOPIVOT if normal cholesky factorization wanted.
 * @param[in] flags
 *      The matrix structure indicator, *ARMAS_UPPER* for upper tridiagonal
 *      and  *ARMAS_LOWER* for lower tridiagonal matrix.
 * @param[in,out] wb
 *      Workspace for pivoting factorization. If wb.bytes is zero then work
 *      buffer size is returned in wb.bytes.
 * @param[in,out] cf
 *      Optional blocking configuration. If not provided default blocking
 *      configuration  will be used.
 *
 * @retval  0 Success; If pivoting factorized then result matrix is full rank.
 * @retval >0 Success with pivoting factorization, matrix rank returned.
 * @retval <0 Failure, _conf.error_ holds error code
 *
 * Pivoting factorization is computed of P is not ARMAS_NOPIVOT. Pivoting
 * factorization needs workspace of size N elements for blocked version.
 * If no workspace (ARMAS_NOWORK) is provided  or it is too small then unblocked
 * algorithm is used.
 *
 * Factorization stops when diagonal element goes small enough.
 * Default value for stopping criteria is \f$ max |diag(A)|*N*epsilon \f$
 * If value of absolute stopping criteria _conf.stop_ is non-zero then it is
 * used. Otherwise if _conf.smult_ (relative stopping criterion multiplier) is
 * non-zero then stopping criteria is set to \f$ max |diag(A)|*smult \f$.
 *
 * Pivoting factorization returns zero if result matrix is full rank.
 * Return value greater than zero is rank of result matrix.
 * Negative values indicate error.
 *
 * Compatible with lapack.DPOTRF
 * @ingroup lapack
 */
int armas_x_cholfactor_w(armas_x_dense_t * A, armas_pivot_t * P,
                         int flags, armas_wbuf_t * wb, armas_conf_t * cf)
{
    armas_x_dense_t W;
    armas_env_t *env;
    int err = 0;
    if (!cf)
        cf = armas_conf_default();

    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    env = armas_getenv();
    if (P != ARMAS_NOPIVOT) {
        if (wb && wb->bytes == 0) {
            wb->bytes = A->cols * sizeof(DTYPE);
            return 0;
        }
        // working space is N elements for blocked factorization
        if (env->lb > 0 && A->cols > env->lb) {
            if (!wb || armas_wbytes(wb) < A->cols * sizeof(DTYPE))
                armas_x_make(&W, 0, 0, 0, (DTYPE *) 0);
            else
                armas_x_make(&W, A->cols, 1, A->cols,
                             (DTYPE *) armas_wptr(wb));
        } else {
            // force unblocked with zero sized workspace
            armas_x_make(&W, 0, 0, 0, (DTYPE *) 0);
        }
        return armas_x_cholfactor_pv(A, &W, P, flags, cf);
    }

    if (A->rows != A->cols) {
        cf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (env->lb == 0 || A->cols <= env->lb) {
        if (flags & ARMAS_LOWER) {
            err = unblk_cholfactor_lower(A, cf);
        } else {
            err = unblk_cholfactor_upper(A, cf);
        }
    } else {
        if (flags & ARMAS_LOWER) {
            err = blk_cholfactor_lower(A, env->lb, cf);
        } else {
            err = blk_cholfactor_upper(A, env->lb, cf);
        }
    }
    return err;
}

/**
 * @brief Solve symmetric positive definite system of linear equations
 *
 * Solves a system of linear equations \f$ AX = B \f$ with symmetric positive
 * definite matrix A using the Cholesky factorization
 * \f$ A = U^TU \f$ or \f$ A = LL^T \f$ computed by `cholfactor()`.
 *
 * @param[in,out] B
 *      On entry, the right hand side matrix B. On exit, the solution matrix X.
 * @param[in] A
 *      The triangular factor U or L from Cholesky factorization as computed by
 *      `cholfactor().`
 * @param[in] P
 *      Optional pivot array. If non null then A is pivoted cholesky
 *      factorization. Set to ARMAS_NOPIVOT if normal cholesky
 *       factorization used.
 * @param[in] flags
 *      Indicator of which factor is stored in A. If *ARMAS_UPPER*
 *       (*ARMAS_LOWER) then upper (lower) triangle of A is stored.
 * @param[in,out] cf
 *       Optional blocking configuration.
 *
 * @retval  0 Succes
 * @retval -1 Error, `conf.error` holds last error code
 *
 * Compatible with lapack.DPOTRS.
 * @ingroup lapack
 */
int armas_x_cholsolve(armas_x_dense_t * B,
                      const armas_x_dense_t * A,
                      const armas_pivot_t * P,
                      int flags, armas_conf_t * cf)
{
    int ok;
    if (!cf)
        cf = armas_conf_default();

    if (P != ARMAS_NOPIVOT) {
        return armas_x_cholsolve_pv(B, (armas_x_dense_t *) A,
                              (armas_pivot_t *) P, flags, cf);
    }

    ok = B->rows == A->cols && A->rows == A->cols;
    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }

    if (flags & ARMAS_LOWER) {
        // solve A*X = B; X = A.-1*B == (L*L.T).-1*B == L.-T*(L.-1*B)
        armas_x_solve_trm(B, ONE, A, ARMAS_LEFT|ARMAS_LOWER, cf);
        armas_x_solve_trm(B, ONE, A, ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANSA, cf);
    } else {
        // solve A*X = B;  X = A.-1*B == (U.T*U).-1*B == U.-1*(U.-T*B)
        armas_x_solve_trm(B, ONE, A, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSA, cf);
        armas_x_solve_trm(B, ONE, A, ARMAS_LEFT|ARMAS_UPPER, cf);
    }
    return 0;
}

#endif	/* ARMAS_PROVIDES && ARMAS_REQUIRES */
