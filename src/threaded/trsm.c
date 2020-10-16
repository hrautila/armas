
// Copyright by libARMAS authors. See AUTHORS file in this archive.

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
#if defined(armas_solve_trm_unsafe)
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
    armas_dense_t B;
    struct armas_ac_block *args = (struct armas_ac_block *)argptr;
    struct armas_ac_blas3 *blas = args->u.blas3;

    /* Slice proper part of argument matrices. */
    switch (blas->flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        armas_submatrix_unsafe(
            &B, blas->B, args->row, 0, args->nrows, blas->B->cols);
        break;
    case ARMAS_LEFT:
    default:
        armas_submatrix_unsafe(
            &B, blas->B, 0, args->column, blas->B->rows, args->ncolumns);
        break;
    }
    armas_cbuf_t *cbuf = armas_cbuf_get_thread_global();
    if (!cbuf) {
        args->error = ARMAS_EMEMORY;
        return (void *)0;
    }

    cache_t cache;
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    armas_solve_trm_unsafe(
        &B, blas->alpha, blas->A, blas->flags, &cache);

    if (!args->is_last)
        armas_cbuf_release_thread_global();
    return (void *)0;
}

static
int threaded_solve_trm_recursive(
    int thread,
    int nthreads,
    struct armas_ac_blas3 *blas,
    armas_conf_t *cf)
{
    int err;
    pthread_t th;
    struct armas_ac_block args;

    if ((blas->flags & ARMAS_RIGHT) != 0) {
        args.row = armas_ac_block_index(thread, nthreads, blas->B->rows);
        args.nrows =
            armas_ac_block_index(thread+1, nthreads, blas->B->rows) - args.row;
        args.column = 0;
        args.ncolumns = blas->B->cols;
    } else {
        args.column = armas_ac_block_index(thread, nthreads, blas->B->cols);
        args.ncolumns =
            armas_ac_block_index(thread+1, nthreads, blas->B->cols) - args.column;
        args.row = 0;
        args.nrows = blas->B->rows;
    }
    blas->C = (armas_dense_t *)0;
    args.u.blas3 = blas;
    args.block_index = thread;
    args.is_last = thread+1 == nthreads;
    args.error = 0;

    /* last block directly in main thread. */
    if (thread+1 == nthreads) {
        compute_block(&args);
        return -args.error;
    }

    err = pthread_create(&th, NULL, compute_block, &args);
    if (err) {
        cf->error = -err;
        return -1;
    }
    err = threaded_solve_trm_recursive(thread+1, nthreads, blas, cf);

    pthread_join(th, NULL);
    return err;
}

int armas_ac_threaded_solve_trm(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf)
{
    size_t nproc = armas_ac_threaded_cores(armas_size(args->B));
    int rc = threaded_solve_trm_recursive(0, nproc, args, cf);
    return rc;
}
#else
#warning "Missing defines. No code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
