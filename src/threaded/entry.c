// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <string.h>
#include <unistd.h>

#include "dtype.h"
#include "armas.h"
#include "matrix.h"
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

static struct armas_ac_vtable vtable = (struct armas_ac_vtable){
    .dispatch = armas_ac_threaded_dispatch,
    .release = armas_ac_threaded_release
};


size_t armas_ac_threaded_cores(size_t num_items)
{
    struct armas_ac_env *env = armas_ac_getenv();
    size_t k = (size_t)ceil((double)num_items/(double)env->num_items);
    return k < env->max_cores ? k : env->max_cores;
}

int armas_ac_threaded_init(struct armas_ac_vtable **vptr, void **private)
{
    *vptr = &vtable;
    *private = (void *)armas_ac_getenv();
    return 0;
}
