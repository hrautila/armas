// Copyright (c) Harri Rautila, 2019-2020

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <string.h>
#include <unistd.h>

#include "dtype.h"
#include "armas.h"
#include "matrix.h"
#include "accel.h"
#include "scheduler.h"
#include "workers.h"

static
int armas_ac_workers_dispatch(int opcode, void *args, armas_conf_t *cf, void *private)
{
    struct armas_ac_workers *wcf = (struct armas_ac_workers *)private;
    int rc = -ARMAS_EIMP;

    switch (opcode) {
    case ARMAS_AC_GEMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        rc = armas_ac_workers_mult((struct armas_ac_blas3 *)args, cf, wcf);
        break;
    case ARMAS_AC_SYMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        //rc = armas_ac_workers_mult_sym((struct armas_ac_blas3 *)args, cf, wcf);
        break;
    case ARMAS_AC_TRMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        //rc = armas_ac_workers_mult_trm((struct armas_ac_blas3 *)args, cf, wcf);
        break;
    case ARMAS_AC_TRSM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        //rc = armas_ac_workers_solve_trm((struct armas_ac_blas3 *)args, cf, wcf);
        break;
    default:
        rc = -ARMAS_EIMP;
        break;
    }
final_exit:
    return rc;
}

static
int armas_ac_workers_release(void *private)
{
    struct armas_ac_workers *config = (struct armas_ac_workers *)private;
    if (!config)
        return 0;
    armas_sched_stop(&config->sched);
    return 0;
}


static struct armas_ac_vtable vtable = {
    .dispatch = armas_ac_workers_dispatch,
    .release = armas_ac_workers_release
};

#if 0
size_t armas_ac_workers_cores(struct armas_workers_env *env, size_t num_items)
{
    size_t k = (size_t)ceil((double)num_items / (double)env->items_per_thread);
    return k < env->max_cores ? k : env->max_cores;
}
#endif

int armas_ac_workers_init(struct armas_ac_vtable **vptr, void **private)
{
    // create scheduler and workers
    struct armas_ac_workers *config =
        (struct armas_ac_workers *)calloc(1, sizeof(struct armas_ac_workers));
    if (!config)
        return -ARMAS_EMEMORY;

    int rc = armas_sched_conf(&config->sched, DEFAULT_QLEN);
    if (rc < 0) {
        free(config);
        return rc;
    }
    *vptr = &vtable;
    *private = (void *)config;
    return 0;
}

