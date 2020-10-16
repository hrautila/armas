
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication

//! \cond
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
//! \endcond

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#define ARMAS_PROVIDES 1
// this file requires external public functions
#if defined(armas_mult_kernel)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "threaded.h"
//! \endcond

static
void *compute_block(void *argptr)
{
    armas_dense_t C, A, B;
    struct armas_ac_block *args = (struct armas_ac_block *)argptr;
    struct armas_ac_blas3 *blas = args->u.blas3;

    /*
     * Slice proper part of argument matrices.
     */
    armas_submatrix_unsafe(
        &C, blas->C, args->row, args->column, args->nrows, args->ncolumns);

    switch (blas->flags & (ARMAS_TRANSA | ARMAS_TRANSB)) {
    case ARMAS_TRANSA | ARMAS_TRANSB:
        armas_submatrix_unsafe(
            &A, blas->A, 0, args->row, blas->A->rows, args->nrows);
        armas_submatrix_unsafe(
            &B, blas->B, args->column, 0, args->ncolumns, blas->B->cols);
        break;
    case ARMAS_TRANSA:
        armas_submatrix_unsafe(
            &A, blas->A, 0, args->row, blas->A->rows, args->nrows);
        armas_submatrix_unsafe(
            &B, blas->B, 0, args->column, blas->B->rows, args->ncolumns);
        break;
    case ARMAS_TRANSB:
        armas_submatrix_unsafe(
            &A, blas->A, args->row, 0, args->nrows, blas->A->cols);
        armas_submatrix_unsafe(
            &B, blas->B, args->column, 0, args->ncolumns, blas->B->cols);
        break;
    default:
        armas_submatrix_unsafe(
            &A, blas->A, args->row, 0, args->nrows, blas->A->cols);
        armas_submatrix_unsafe(
            &B, blas->B, 0, args->column, blas->B->rows, args->ncolumns);
        break;
    }
    cache_t cache;
    struct armas_cbuf *cbuf = armas_cbuf_get_thread_global();
    if (!cbuf) {
        args->error = ARMAS_EMEMORY;
        return (void *)0;
    }

    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    armas_mult_kernel(blas->beta, &C,
                        blas->alpha, &A, &B, blas->flags, &cache);

    return (void *)0;
}

// Compute recursively in nthreads threads.
static
int threaded_mult_recursive(
    int thread,
    int nthreads,
    struct armas_ac_blas3 *blas,
    armas_conf_t *cf)
{
    int err;
    pthread_t th;
    struct armas_ac_block args;

    //printf("mult_recursive %d/%d\n", thread, nthreads);
    if (blas->C->rows <= blas->C->cols) {
        args.column = armas_ac_block_index(thread, nthreads, blas->C->cols);
        args.ncolumns =
            armas_ac_block_index(thread+1, nthreads, blas->C->cols) - args.column;
        args.row = 0;
        args.nrows = blas->C->rows;
    } else {
        args.row = armas_ac_block_index(thread, nthreads, blas->C->rows);
        args.nrows =
            armas_ac_block_index(thread+1, nthreads, blas->C->rows) - args.row;
        args.column = 0;
        args.ncolumns = blas->C->cols;
    }
    args.u.blas3 = blas;
    args.block_index = thread;
    args.is_last = thread+1 == nthreads;
    args.error = 0;

    // last block
    if (thread+1 == nthreads) {
        compute_block(&args);
        return -args.error;
    }

    //printf("create thread %d\n", thread);
    err = pthread_create(&th, NULL, compute_block, &args);
    if (err) {
        cf->error = -err;
        return -1;
    }
    err = threaded_mult_recursive(thread+1, nthreads, blas, cf);

    //printf("wait thread %d\n", thread);
    pthread_join(th, NULL);
    return err;
}

int armas_ac_threaded_mult(
    struct armas_ac_blas3 *args,
    armas_conf_t *cf,
    struct armas_threaded_conf *acf)
{
    size_t nproc = armas_ac_threaded_cores(armas_size(args->C));
    int rc = threaded_mult_recursive(0, nproc, args, cf);
    return rc;
}
#else
#warning "Missing defines. No code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
