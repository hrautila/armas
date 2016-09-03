
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_xyz) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_wzx) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

// template function with matrix partitioning from top-left to bottom-right
static
int __local(armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
    // full partitioning; diagonal elements are required, others only as needed
    // (replace with __nil if not needed)
    armas_x_dense_t ATL, ABL, ABR, ATR, A00, a01, A02, a10, a11, a12, A20, a21, A22;
    int err = 0;
    DTYPE a11val;

    __partition_2x2(&ATL, &ATR,
                    &ABL, &ABR,   /**/  A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        __repartition_2x2to3x3(&ATL,
                               &A00, &a01, &A02,
                               &a10, &a11, &a12,
                               &A20, &a21, &A22,  /**/  A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------------

        // ---------------------------------------------------------------------------
        __continue_3x3to2x2(&ATL, &ATR,
                            &ABL, &ABR, /**/  &A00, &a11, &A22,   A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

