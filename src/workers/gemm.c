
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication

//! \cond
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
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
#include "accel.h"
#include "scheduler.h"
#include "counter.h"
#include "workers.h"
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
    struct armas_cbuf *cbuf = armas_cbuf_create_thread_global();
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    armas_mult_kernel(blas->beta, &C,
                        blas->alpha, &A, &B, blas->flags, &cache);

    return (void *)0;
}

static inline
void armas_ac_define_block(
    struct armas_ac_block *blk,
    int index,
    int nblk,
    int major,
    int minor,
    int colwise)
{
    if (colwise) {
        blk->row = 0;
        blk->nrows = minor;
        blk->column = armas_ac_block_index(index, nblk, major);
        blk->ncolumns =
            armas_ac_block_index(index+1, nblk, major) - blk->column;
    } else {
        blk->column = 0;
        blk->ncolumns = minor;
        blk->row = armas_ac_block_index(index, nblk, major);
        blk->nrows =
            armas_ac_block_index(index+1, nblk, major) - blk->row;
    }
}

static
void schedule_blocks(
    struct armas_ac_blas3 *args,
    struct armas_ac_worker_task *tasks,
    int ntask,
    struct armas_ac_counter *ready,
    struct armas_ac_scheduler *scheduler)
{
    struct armas_ac_block *blkargs;
    int colwise = args->C->cols > args->C->rows;
    int minor = colwise ? args->C->rows : args->C->cols;
    int major = colwise ? args->C->cols : args->C->rows;

    for (int k = 0; k < ntask; k++) {
        blkargs = &tasks[k].args;
        armas_ac_define_block(blkargs, k, ntask, major, minor, colwise);

        blkargs->u.blas3 = args;
        blkargs->block_index = k;
        blkargs->is_last = k + 1 == ntask;

        armas_task_init(&tasks[k].task, k, compute_block, blkargs, ready);
        armas_sched_schedule(scheduler, &tasks[k]);
    }
}

static
void schedule_tiles(
    struct armas_ac_blas3 *args,
    struct armas_ac_worker_task *tasks,
    int ntask,
    int rtiles,
    int ctiles,
    struct armas_ac_counter *ready,
    struct armas_ac_scheduler *scheduler)
{
    int k, j, i, cstart, ccount, rstart, rcount;
    struct armas_ac_block *blkargs;

    for (j = 0, k = 0; j < ctiles; j++) {
        cstart = armas_ac_block_index(j, ctiles, args->C->cols);
        ccount = armas_ac_block_index(j + 1, ctiles, args->C->cols) - cstart;

        for (i = 0; i < rtiles; i++) {
            blkargs = &tasks[k].args;
            blkargs->column = cstart;
            blkargs->ncolumns = ccount;
            blkargs->u.blas3 = args;

            rstart = armas_ac_block_index(i, rtiles, args->C->rows);
            rcount =
                armas_ac_block_index(i + 1, rtiles, args->C->rows) - rstart;
            // set parameters
            blkargs->row = rstart;
            blkargs->nrows = rcount;

            blkargs->u.blas3 = args;
            blkargs->block_index = k;
            blkargs->is_last = k + 1 == ntask;
            armas_task_init(&tasks[k].task, k, compute_block, blkargs, ready);
            armas_sched_schedule(scheduler, &tasks[k]);
            k++;
        }
    }
}

int armas_ac_workers_mult(
    struct armas_ac_blas3 *args,
    armas_conf_t *cf,
    struct armas_ac_scheduler *scheduler)
{
    int rN = 0, cN = 0;
    size_t ntask;
    struct armas_ac_worker_task *tasks;
    struct armas_ac_counter ready;
    struct armas_ac_env *env = armas_ac_getenv();

    if (env->options & ARMAS_OBLAS_TILED) {
        // wb = ((int)floor(sqrt((double)env->num_items))) & 0x3;
        ntask = worker_tiles(&rN, &cN, args->C->rows, args->C->cols, env->num_items);
    } else {
        ntask = (int)(env->weight*armas_size(args->C)/env->num_items);
        if (ntask > env->max_cores)
            ntask = env->max_cores;
    }
    /* Small matrix;
     * Return not implemented and use basic unthreaded implementation.
     */
    if (ntask == 0)
        return -ARMAS_EIMP;

    tasks = (struct armas_ac_worker_task *)calloc(ntask, sizeof(struct armas_ac_worker_task));
    if (!tasks) {
        cf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    armas_counter_init(&ready, ntask);

    if (env->options & ARMAS_OBLAS_TILED) {
        schedule_tiles(args, tasks, ntask, rN, cN, &ready, scheduler);
    } else {
        schedule_blocks(args, tasks, ntask, &ready, scheduler);
    }

    // wait for tasks to finish
    armas_counter_wait(&ready);
    // verify that task worker count is zero on all tasks
    int refcnt = worker_all_ready(tasks, ntask);
    assert(refcnt == 0);
    // release task memory
    free(tasks);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */
