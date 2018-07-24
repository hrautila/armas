
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_icholz) && defined(armassp_x_init_icholz)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_mvsolve_trm) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <math.h>
#include <assert.h>
#include "matrix.h"
#include "sparse.h"

#include "splocal.h"

// reference:
//   (1) Saad, Iterative Methods for Sparse Linear Systems, 2nd Ed.

/*
 * \brief Incomplete Cholesky preconditioner

     l11 |  0    l11 | l21^T      a11 |  0
     ----+----   ----+------  ==  ----+----
     l21 | L22    0  | L22^T      a21 | A22

     l11 = sqrt(a11)
     l21 = a21/a11
     A22 = A22 - l21*l21^T  (A22 lower triangular)

     u11 |   0     u11 | u12^T      a11 | a12^T
     ----+------   ----+------  ==  ----+------
     u12 | U22^T    0  | U22         0  | A22

     u11 = sqrt(a11)
     u12^T = a12^T/a11
     A22 = A22 - u12*u12^T  (A22 upper triangular)

 */

static
int csr_illtz_lower(armas_x_sparse_t *A)
{
    int i, k, j, p, p0, p1, p2;
    DTYPE *Le, a11;

    Le = A->elems.v;
    
    // compute l11 = sqrt(a11); l21 = a21/l11; A22 = A22 - l21*l21^T
    for (i = 0; i < A->rows; i++) {
        for (p0 = armassp_x_index(A, i); armassp_x_at(A, p0) < i; p0++);
        // assert p0 points to diagonal entry; p0+1 start of a12^T
        assert(armassp_x_at(A, p0) == i);

        a11 = sqrt(armassp_x_value(A, p0));
        Le[p0] = a11;      
        // l21 = a21/a11; A22 = A22 - l21*l21^T
        for (k = i+1; k < A->rows; k++) {
            // if A[k,i] == 0 continue
            if ((p1 = armassp_x_nz(A, k, i)) < 0)
                continue;
            // l21[k] = a21[k]/a11
            Le[p1] /= a11;
            // A22[k,:] = A22[k,:] - l21[k]*l21^T[0:k]
            for (p = p1+1; armassp_x_at(A, p) <= k && p < armassp_x_index(A, k+1); p++) {
                // here j <= k
                j = armassp_x_at(A, p);
                if ((p2 = armassp_x_nz(A, j, i)) < 0)
                    continue;
                // here p2 <= p1
                Le[p] -= Le[p1]*Le[p2];
            }
        }
    }
    //A->nnz = A->nnz;
    return 0;
}

static
int csr_illtz_upper(armas_x_sparse_t *A)
{
    int i, j, p, p0, p1, p2;
    DTYPE *Ue, a11;

    Ue = A->elems.v;
    
    // compute u11 = sqrt(a11); u12 = a12/u11; A22 = A22 - u12*u12^T
    for (i = 0; i < A->rows; i++) {
        for (p0 = armassp_x_index(A, i); armassp_x_at(A, p0) < i; p0++);
        // assert p0 points to diagonal entry; p0+1 start of a12^T
        assert(armassp_x_at(A, p0) == i);

        a11 = sqrt(armassp_x_value(A, p0));
        Ue[p0] = a11;      

        // u12^T = u12^T/a11;
        for (p = p0+1; p < armassp_x_index(A, i+1); p++)
            Ue[p] /= a11;

        // A22 = A22 - u12*u12^T
        for (p = p0+1; p < armassp_x_index(A, i+1); p++) {
            // j'th column on this column ; j > i;
            j = armassp_x_at(A, p);
            // find diagonal
            for (p2 = armassp_x_index(A, j); armassp_x_at(A, p2) < j && p2 < armassp_x_index(A, j+1); p2++);
            // update values on and above diagonal
            for (p1 = p0+1; p1 < armassp_x_index(A, i+1) && p2 < armassp_x_index(A, j+1); ) {
                int d = armassp_x_at(A, p1) - armassp_x_at(A, p2);
                if (d == 0) {
                    Ue[p2] -= Ue[p1]*Ue[p];
                    p1++; p2++;
                } else if (d < 0) {
                    p1++;
                } else {
                    p2++;
                }
            }
        }
    }
    return 0;
}    


static
int csc_illtz_lower(armas_x_sparse_t *A)
{
    int i, j, p, p0, p1, p2;
    DTYPE *Le, a11;

    Le = A->elems.v;
    
    // compute l11 = sqrt(a11); l21 = a21/l11; A22 = A22 - l21*l21^T
    for (i = 0; i < A->cols; i++) {
        for (p0 = armassp_x_index(A, i); armassp_x_at(A, p0) < i; p0++);
        // assert p0 points to diagonal entry; p0+1 start of a12^T
        assert(armassp_x_at(A, p0) == i);

        a11 = sqrt(armassp_x_value(A, p0));
        Le[p0] = a11;      
        // l21 = a21/a11;
        for (p = p0+1; p < armassp_x_index(A, i+1); p++)
            Le[p] /= a11;
        
        // A22 = A22 - l21*l21^T
        for (p = p0+1; p < armassp_x_index(A, i+1); p++) {
            // j'th row on this column ; j > i;
            j = armassp_x_at(A, p);
            // find diagonal
            for (p2 = armassp_x_index(A, j); armassp_x_at(A, p2) < j && p2 < armassp_x_index(A, j+1); p2++);
            // update values on and below diagonal
            for (p1 = p0+1; p1 < armassp_x_index(A, i+1) && p2 < armassp_x_index(A, j+1); ) {
                int d = armassp_x_at(A, p1) - armassp_x_at(A, p2);
                if (d == 0) {
                    Le[p2] -= Le[p1]*Le[p];
                    p1++; p2++;
                } else if (d < 0) {
                    p1++;
                } else {
                    p2++;
                }
            }
        }
        
    }
    return 0;
}

static
int csc_illtz_upper(armas_x_sparse_t *A)
{
    int i, k, j, p, p0, p1, p2;
    DTYPE *Ue, a11;

    Ue = A->elems.v;
    
    // compute l11 = sqrt(a11); l21 = a21/l11; A22 = A22 - l21*l21^T
    for (i = 0; i < A->rows; i++) {
        for (p0 = armassp_x_index(A, i); armassp_x_at(A, p0) < i; p0++);
        // assert p0 points to diagonal entry; p0+1 start of a12^T
        assert(armassp_x_at(A, p0) == i);

        a11 = sqrt(armassp_x_value(A, p0));
        Ue[p0] = a11;      
        // u12^T = u12^T/a11; A22 = A22 - u12*u12^T
        for (k = i+1; k < A->cols; k++) {
            // if A[k,i] == 0 continue
            if ((p1 = armassp_x_nz(A, k, i)) < 0)
                continue;
            // u12^T[k] = a12^T[k]/a11
            Ue[p1] /= a11;
            // A22[:,k] = A22[:,k] - u12[k]*u12^T[0:k]
            for (p = p1+1; armassp_x_at(A, p) <= k && p < armassp_x_index(A, k+1); p++) {
                // here j <= k
                j = armassp_x_at(A, p);
                if ((p2 = armassp_x_nz(A, j, i)) < 0)
                    continue;
                // here p2 <= p1
                Ue[p] -= Ue[p1]*Ue[p2];
            }
        }
    }
    return 0;
}

/**
 */
int armassp_x_icholz(armas_x_sparse_t *A, int flags)
{
    switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {

    case ARMAS_UPPER:
        if (A->kind == ARMASSP_CSR)
            csr_illtz_upper(A);
        else
            csc_illtz_upper(A);          
        break;
    case ARMAS_LOWER:
    default:
        if (A->kind == ARMASSP_CSR) 
            csr_illtz_lower(A);
        else
            csc_illtz_lower(A);
    }
    return 0;
}


static
int __x_precond_icholz(armas_x_dense_t *z,
                       const armassp_x_precond_t *P,
                       const armas_x_dense_t *x,
                       armas_conf_t *cf)
{
    if (z != x) {
        armas_x_mcopy(z, x);
    }
    if ((P->flags & ARMAS_UPPER) != 0) {
        armassp_x_mvsolve_trm(z, 1.0, P->M, P->flags|ARMAS_TRANS, cf);
        armassp_x_mvsolve_trm(z, 1.0, P->M, P->flags, cf);
    } else {
        // x = M^-1*x = (LL^T)^-1*x = L^-T*(L^-1*x)
        armassp_x_mvsolve_trm(z, 1.0, P->M, P->flags, cf);
        armassp_x_mvsolve_trm(z, 1.0, P->M, P->flags|ARMAS_TRANS, cf);
    }
    return 0;
}

static
int __x_precond_icholz_partial(armas_x_dense_t *z,
                               const armassp_x_precond_t *P,
                               const armas_x_dense_t *x,
                               int flags,
                               armas_conf_t *cf)
{
    if (z != x) {
        armas_x_mcopy(z, x);
    }
    armassp_x_mvsolve_trm(z, 1.0, P->M, P->flags|flags, cf);
    return 0;
}

/**
 */
int armassp_x_init_icholz(armassp_x_precond_t *P, armas_x_sparse_t *A, int flags)
{
    int stat = armassp_x_icholz(A, flags);
    if (stat == 0) {
        P->M = A;
        P->precond = __x_precond_icholz;
        P->partial = __x_precond_icholz_partial;
        P->flags = flags;
    }
    return stat;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
