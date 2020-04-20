
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

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
#if defined(armas_x_solve_trm_unsafe)
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
    armas_x_dense_t A, B;
    struct threaded_block_args *args = (struct threaded_block_args *)argptr;
    struct armas_ac_blas3 *blas = args->u.blas3;
    int nrows = args->lrow - args->frow;
    int ncols = args->lcol - args->fcol;

    //printf("compute block: [%d,%d] [%d,%d]\n", args->frow, args->fcol, args->lrow, args->lcol);
    switch (blas->flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        armas_x_submatrix_unsafe(&B, blas->B, args->fcol, 0, nrows, blas->B->cols);
        break;
    case ARMAS_LEFT:
    default:
        armas_x_submatrix_unsafe(&B, blas->B, 0, args->fcol, blas->B->rows, ncols);
        break;
    }
    armas_cbuf_t *cbuf = armas_cbuf_create_thread_global();

    cache_t cache;
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    armas_x_solve_trm_unsafe(&B, blas->alpha, &A, blas->flags, &cache);

    // Last block is computed in main thread. Do not release thread global resources.
    if (!args->last)
        armas_cbuf_release_thread_global();
    return (void *)0;
}


// Compute recursively in nthreads threads.
static
int threaded_solve_trm_recursive(int thread, int nthreads, struct armas_ac_blas3 *blas, armas_conf_t *cf)
{
    int first_col, last_col, first_row, last_row, err;
    pthread_t th;
    struct threaded_block_args args;

    //printf("mult_recursive %d/%d\n", thread, nthreads);
    if ((blas->flags & ARMAS_RIGHT) != 0) {
        first_row = block_index4(thread, nthreads, blas->B->rows);
        last_row  = block_index4(thread+1, nthreads, blas->B->rows);
        first_col = 0; last_col = blas->C->cols;
    } else {
        first_col = block_index4(thread, nthreads, blas->B->cols);
        last_col  = block_index4(thread+1, nthreads, blas->B->cols);
        first_row = 0; last_row = blas->C->rows;
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
    err = threaded_solve_trm_recursive(thread+1, nthreads, blas, cf);

    //printf("wait thread %d\n", thread);
    pthread_join(th, NULL);
    return err;
}

int armas_ac_threaded_solve_trm(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf)
{
    size_t nproc = armas_ac_threaded_cores(armas_x_size(args->B));
    int rc = threaded_solve_trm_recursive(0, nproc, args, cf);
    return rc;
}
#else
#warning "Missing defines. No code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
