// Copyright (c) Harri Rautila, 2019-2020

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <string.h>
#include <unistd.h>

#include "armas.h"
#include "accel.h"
#include "threaded.h"

#ifndef ENV_ARMAS_THREADED
#define ENV_ARMAS_THREADED "ARMAS_THREADED"
#endif

static
int armas_ac_threaded_dispatch(int opcode, void *args, armas_conf_t *cf, void *private)
{
    int rc;
    struct armas_threaded_conf *acf = (struct armas_threaded_conf *)private;

    switch (opcode) {
    case ARMAS_AC_GEMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        rc = armas_ac_threaded_mult((struct armas_ac_blas3 *)args, cf, acf);
        break;
    case ARMAS_AC_SYMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        rc = armas_ac_threaded_mult_sym((struct armas_ac_blas3 *)args, cf, acf);
        break;
    case ARMAS_AC_TRMM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        rc = armas_ac_threaded_mult_trm((struct armas_ac_blas3 *)args, cf, acf);
        break;
    case ARMAS_AC_TRSM:
        if (!armas_ac_test_tag(args, ARMAS_AC_BLAS3)) {
            rc = -ARMAS_EINVAL;
            goto final_exit;
        }
        rc = armas_ac_threaded_solve_trm((struct armas_ac_blas3 *)args, cf, acf);
        break;
    default:
        rc = -ARMAS_EIMP;
        break;
    }
final_exit:
    return rc;
}

static
int armas_ac_threaded_release(void *private)
{
    return 0;
}

static
void threaded_config(struct armas_threaded_conf *acf)
{
    int n, val;
    char *tok, *cstr;

    acf->max_cores = sysconf(_SC_NPROCESSORS_ONLN);
    cstr = getenv(ENV_ARMAS_THREADED);
    // parse string: "ELEMS_PER_THREAD,MAXPROCESSORS"
    for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
        val = atoi(tok);
        switch (n) {
        case 0:
            if (val > 0)
                acf->items_per_thread = val;
            break;
        case 1:
            if (val > 0 && val < acf->max_cores)
                acf->max_cores = val;
            break;
        default:
            break;
        }
    }
}

static struct armas_ac_vtable vtable = (struct armas_ac_vtable){
    .dispatch = armas_ac_threaded_dispatch,
    .release = armas_ac_threaded_release
};

static struct armas_threaded_conf acf = (struct armas_threaded_conf){
    .max_cores = 1,
    .items_per_thread = 400*400
};

size_t armas_ac_threaded_cores(struct armas_threaded_conf *acf, size_t num_items)
{
    size_t k = (size_t)ceil((double)num_items/(double)acf->items_per_thread);
    return k < acf->max_cores ? k : acf->max_cores;
}

int armas_ac_threaded_init(struct armas_ac_vtable **vptr, void **private)
{
    threaded_config(&acf);
    *vptr = &vtable;
    *private = (void *)&acf;
    return 0;
}
