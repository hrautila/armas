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
    struct armas_ac_scheduler *scheduler = (struct armas_ac_scheduler *)private;
    int rc = -ARMAS_EIMP;

    switch (opcode) {
    case ARMAS_AC_GEMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        rc = armas_ac_workers_mult((struct armas_ac_blas3 *)args, cf, scheduler);
        break;
    case ARMAS_AC_SYMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        //rc = armas_ac_workers_mult_sym((struct armas_ac_blas3 *)args, cf, scheduler);
        break;
    case ARMAS_AC_TRMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        //rc = armas_ac_workers_mult_trm((struct armas_ac_blas3 *)args, cf, scheduler);
        break;
    case ARMAS_AC_TRSM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        //rc = armas_ac_workers_solve_trm((struct armas_ac_blas3 *)args, cf, scheduler);
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
    struct armas_ac_scheduler *scheduler = (struct armas_ac_scheduler *)private;
    if (!scheduler)
        return 0;
    armas_sched_stop(scheduler);
    return 0;
}

static struct armas_ac_vtable vtable = {
    .dispatch = armas_ac_workers_dispatch,
    .release = armas_ac_workers_release
};


int armas_ac_workers_init(struct armas_ac_vtable **vptr, void **private, int transient)
{
    int rc;
    // create scheduler and workers
    struct armas_ac_scheduler *scheduler = (struct armas_ac_scheduler *)0;

    if (transient) {
        rc = armas_ac_sched_transient_init(&scheduler, DEFAULT_QLEN);
    } else {
        rc = armas_ac_sched_workers_init(&scheduler, DEFAULT_QLEN);
    }
    if (rc < 0)
        return rc;

    *vptr = &vtable;
    *private = (void *)scheduler;
    return 0;
}
