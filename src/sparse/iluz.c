
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_iluz) && defined(armassp_init_iluz)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_blas) && defined(armassp_core)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include <assert.h>
#include "matrix.h"
#include "sparse.h"

// reference:

/*
    ILU(0) in IKJ formulation
    --------------------------
    for i = 2 ... n:
      for k = 1 ... i-1 and (i,k) in nz(A):
         a_ik = a_ik/a_kk
         for j = k+1 ... n and (i,j) in nz(A):
           a_ij = a_ij - a_ik*a_kj
         end
      end
    end

      1  |  0    u11 | u12      a11 | a12^T
     ----+----   ----+----  ==  ----+------
     l21 | L22    0  | U22      a21 | A22

     u11 = a11
     u12 = a12
     l21 = a21/a11
     A22 = A22 - l21*a12^T
 */

static
int csr_iluz(armas_sparse_t * L)
{
    int i, k, p, u, d, p0, p1;
    DTYPE *Le, a11;

    Le = L->elems.v;

    // IKJ loop; compute l21 = a21/a11; A22 = A22 - l21*a12^T
    for (i = 0; i < L->rows; i++) {
        for (p0 = armassp_index(L, i); armassp_at(L, p0) < i; p0++);
        // assert p0 points to diagonal entry; p0+1 start of a12^T
        assert(armassp_at(L, p0) == i);
        a11 = armassp_value(L, p0);

        // l21 = a21/a11; A22 = A22 - l21*a12^T
        for (k = i + 1; k < L->rows; k++) {
            // if A[k,i] == 0 continue
            if ((p1 = armassp_nz(L, k, i)) < 0)
                continue;
            // l21[k] = a21[k]/a11
            Le[p1] /= a11;
            // A22[k,:] = A22[k,:] - l21[k]*a12^T[:]
            for (p = p1 + 1, u = p0 + 1;
                 p < armassp_index(L, k + 1)
                 && u < armassp_index(L, i + 1);) {
                d = armassp_at(L, p) - armassp_at(L, u);
                if (d == 0) {
                    Le[p] -= Le[p1] * Le[u];
                    p++;
                    u++;
                } else if (d < 0) {
                    p++;
                } else {
                    u++;
                }
            }
        }
    }
    return 0;
}

static
int csc_iluz(armas_sparse_t * L)
{
    int i, k, p, u, d, p0, p1;
    DTYPE *Le, a11;

    Le = L->elems.v;

    // IKJ loop; compute l21 = a21/a11; A22 = A22 - l21*a12^T
    for (i = 0; i < L->cols; i++) {
        for (p0 = armassp_index(L, i); armassp_at(L, p0) < i; p0++);
        // assert p0 points to diagonal entry; p0+1 start of a21
        assert(armassp_at(L, p0) == i);
        a11 = armassp_value(L, p0);

        // l21 = a21/a11
        for (p = p0 + 1; p < armassp_index(L, i + 1); p++) {
            Le[p] /= a11;
        }

        // L22 = L22 - l21*a12^T
        for (k = i + 1; k < L->cols; k++) {
            // if A[i, k] == 0 continue
            if ((p1 = armassp_nz(L, k, i)) < 0)
                continue;
            // A22[:,k] = A22[:,k] - l21*a12^T[k]
            for (p = p1 + 1, u = p0 + 1;
                 p < armassp_index(L, k + 1)
                 && u < armassp_index(L, i + 1);) {
                d = armassp_at(L, p) - armassp_at(L, u);
                if (d == 0) {
                    Le[p] -= Le[p1] * Le[u];
                    p++;
                    u++;
                } else if (d < 0) {
                    p++;
                } else {
                    u++;
                }
            }
        }
    }
    return 0;
}

int armassp_iluz(armas_sparse_t * L)
{
    if (L->kind != ARMASSP_CSR && L->kind != ARMASSP_CSC)
        return -1;

    if (L->kind == ARMASSP_CSR) {
        return csr_iluz(L);
    }
    return csc_iluz(L);
}

static
int precond_ilu(armas_dense_t * z,
                const armassp_precond_t * P,
                const armas_dense_t * x, armas_conf_t * cf)
{
    if (z != x)
        armas_mcopy(z, x, 0, cf);
    // x = M^-1*x = L^-1*(U^-1*x)
    armassp_mvsolve_trm(z, ONE, P->M, ARMAS_UPPER, cf);
    armassp_mvsolve_trm(z, ONE, P->M, ARMAS_UNIT | ARMAS_LOWER, cf);
    return 0;
}

/**
 * @brief Create incomplete LU preconditioner.
 *
 * @param[in,out] P
 *   On entry uninitialized preconditioner. On exit initialized preconditioner
 *   for matrix.
 * @param[in] A
 *   Sparse matrix. On exit preconditioned matrix.
 *
 * For details see: Saad, *Iterative Methods for Sparse Linear Systems*, 2nd Ed.
 *
 * @retval  0  Success
 */
int armassp_init_iluz(armassp_precond_t * P, armas_sparse_t * A)
{
    int stat = armassp_iluz(A);
    if (stat == 0) {
        P->M = A;
        P->flags = 0;
        P->precond = precond_ilu;
    }
    return stat;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
