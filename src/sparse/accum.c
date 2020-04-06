
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_accum_t)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include <assert.h>
#include "matrix.h"
#include "sparse.h"

// -----------------------------------------------------------------------------
// sparse accumulator

static inline void armas_x_accum_init(armas_x_accum_t * acc, int n, void *ptr)
{
    acc->elems = (DTYPE *) ptr;
    acc->mark = (int *) &acc->elems[n];
    acc->queue = &acc->mark[n];
    acc->nz = n;
    acc->tail = 0;
}

size_t armas_x_accum_make(armas_x_accum_t * acc, int n, void *ptr, size_t len)
{
    size_t nb = armas_x_accum_bytes(n);
    if (nb > len)
        return 0;
    armas_x_accum_init(acc, n, ptr);
    return nb;
}

int armas_x_accum_allocate(armas_x_accum_t * acc, int n)
{
    // single block of n DTYPE and 2*n ints
    size_t nb = armas_x_accum_bytes(n);
    void *ptr = calloc(nb, 1);
    if (!ptr)
        return -1;
    armas_x_accum_init(acc, n, ptr);
    return 0;
}

void armas_x_accum_release(armas_x_accum_t * acc)
{
    if (!acc)
        return;
    free(acc->elems);
    acc->elems = (DTYPE *) 0;
    acc->mark = acc->queue = (int *) 0;
    acc->nz = 0;
}

static inline
void addpos(armas_x_accum_t * acc, int index, DTYPE value, int mark)
{
    if (acc->mark[index] < mark) {
        acc->mark[index] = mark;
        acc->elems[index] = value;
        acc->queue[acc->tail++] = index;
    } else {
        acc->elems[index] += value;
    }
}

// \brief add value to accumulator position
void armas_x_accum_addpos(armas_x_accum_t * acc,
                          int index, DTYPE value, int mark)
{
    addpos(acc, index, value, mark);
}

// accumulate acc = acc + beta*x[:]
void armas_x_accum_scatter(armas_x_accum_t * acc,
                           const armas_x_spvec_t * x, DTYPE beta, int mark)
{
    for (int i = 0; i < x->nz; i++) {
        int k = x->ix[i];
        if (acc->mark[k] < mark) {
            acc->mark[k] = mark;
            acc->elems[k] = beta * x->elems[i];
            acc->queue[acc->tail++] = k;
        } else {
            acc->elems[k] += beta * x->elems[i];
        }
    }
}

void armas_x_accum_dot(armas_x_accum_t * acc,
                       int k,
                       const armas_x_spvec_t * x,
                       const armas_x_spvec_t * y, int mark)
{
    int px, py, d, changed = 0;
    DTYPE v;
    px = py = 0;
    v = ZERO;
    while (px < x->nz && py < y->nz) {
        d = x->ix[px] - y->ix[py];
        if (d == 0) {
            // index match
            v += x->elems[px] * y->elems[py];
            px++;
            py++;
            changed = 1;
        } else {
            d < 0 ? px++ : py++;
        }
    }
    if (changed == 0)
        return;

    addpos(acc, k, v, mark);
}

void armas_x_accum_gather(armas_x_sparse_t * C,
                          DTYPE alpha, armas_x_accum_t * acc, int ik, int maxnz)
{
    int k, head;
    C->ptr[ik] = C->nnz;
    for (head = 0; head < acc->tail; head++) {
        k = acc->queue[head];
        C->ix[C->nnz] = k;
        C->elems.v[C->nnz] = alpha * acc->elems[k];
        C->nnz++;
        // need realloc??; core dump now if yes
        assert(C->nnz < C->size);
    }
}

void armas_x_accum_clear(armas_x_accum_t * acc)
{
    for (int i = 0; i < acc->nz; acc->mark[i++] = -1);
}
#else
#warning "Missing defines. No code!"
#endif  /* ARMAS_PROVIDES && ARMAS_REQUIRES */
