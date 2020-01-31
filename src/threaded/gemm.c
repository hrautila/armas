
// Copyright (c) Harri Rautila, 2012-2014

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
// this file provides following type independet functions
#define ARMAS_PROVIDES 1
// this file requires external public functions
#if defined(armas_x_mult_kernel)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "threaded.h"
//! \endcond


static
void *threaded_compute_block(void *argptr)
{
    armas_x_dense_t C, A, B;
    struct threaded_block_args *args = (struct threaded_block_args *)argptr;
    struct armas_ac_blas3 *blas = args->u.blas3;
    int nrows = args->lrow - args->frow;
    int ncols = args->lcol - args->fcol;

    //printf("compute block: [%d,%d] [%d,%d]\n", args->frow, args->fcol, args->lrow, args->lcol);
    /*
     * Slice proper part of argument matrices.
     */
    armas_x_submatrix_unsafe(&C, blas->C, args->frow, args->fcol, nrows, ncols);

    switch (blas->flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
    case ARMAS_TRANSA|ARMAS_TRANSB:
        armas_x_submatrix_unsafe(&A, blas->A, 0, args->frow, blas->A->rows, nrows);
        armas_x_submatrix_unsafe(&B, blas->B, args->fcol, 0, ncols, blas->B->cols);
        break;
    case ARMAS_TRANSA:
        armas_x_submatrix_unsafe(&A, blas->A, 0, args->frow, blas->A->rows, nrows);
        armas_x_submatrix_unsafe(&B, blas->B, 0, args->fcol, blas->B->rows, ncols);
        break;
    case ARMAS_TRANSB:
        armas_x_submatrix_unsafe(&A, blas->A, args->frow, 0, nrows, blas->A->cols);
        armas_x_submatrix_unsafe(&B, blas->B, args->fcol, 0, ncols, blas->B->cols);
        break;
    default:
        armas_x_submatrix_unsafe(&A, blas->A, args->frow, 0, nrows, blas->A->cols);
        armas_x_submatrix_unsafe(&B, blas->B, 0, args->fcol, blas->B->rows, ncols);
        break;
    }
    armas_cbuf_t *cbuf = armas_cbuf_create_thread_global();

    cache_t cache;
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    armas_x_mult_kernel(blas->beta, &C, blas->alpha, &A, &B, blas->flags, &cache);

    // Last block is computed in main thread. Do not release thread global resources.
    if (!args->last)
        armas_cbuf_release_thread_global();
    return (void *)0;
}


// Compute recursively in nthreads threads.
static
int threaded_mult_recursive(int thread, int nthreads, struct armas_ac_blas3 *blas, armas_conf_t *cf)
{
    int first_col, last_col, first_row, last_row, err;
    pthread_t th;
    struct threaded_block_args args;

    //printf("mult_recursive %d/%d\n", thread, nthreads);
    if (blas->C->rows <= blas->C->cols) {
        first_col = block_index4(thread, nthreads, blas->C->cols);
        last_col  = block_index4(thread+1, nthreads, blas->C->cols);
        first_row = 0; last_row = blas->C->rows;
    } else {
        first_row = block_index4(thread, nthreads, blas->C->rows);
        last_row  = block_index4(thread+1, nthreads, blas->C->rows);
        first_col = 0; last_col = blas->C->cols;
    }

    args.fcol = first_col;
    args.lcol = last_col;
    args.frow = first_row;
    args.lrow = last_row;
    args.u.blas3 = blas;
    args.last = thread+1 == nthreads;

    // last block
    if (thread+1 == nthreads) {
        threaded_compute_block(&args);
        return 0;
    }

    //printf("create thread %d\n", thread);
    err = pthread_create(&th, NULL, threaded_compute_block, &args);
    if (err) {
        cf->error = -err;
        return -1;
    }
    err = threaded_mult_recursive(thread+1, nthreads, blas, cf);

    //printf("wait thread %d\n", thread);
    pthread_join(th, NULL);
    return err;
}

int armas_ac_threaded_mult(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf)
{
    size_t nproc = armas_ac_threaded_cores(acf, armas_x_size(args->C));
    int rc = threaded_mult_recursive(0, nproc, args, cf);
    return rc;
}
#else
#warning "Missing defines. No code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
