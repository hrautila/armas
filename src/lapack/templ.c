
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_xyz)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_wzx)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

// template function with matrix partitioning from top-left to bottom-right
static
int local_func(armas_dense_t * A, int flags, armas_conf_t * cf)
{
    // full partitioning; diagonal elements are required, others only as needed
    // (replace with __nil if not needed)
    armas_dense_t ATL, ABL, ABR, ATR, A00, a01, A02;
    armas_dense_t a10, a11, a12, A20, a21, A22;
    int err = 0;
    DTYPE a11val;

    mat_partition_2x2(
        &ATL, &ATR, &ABL, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------

        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

#endif                          /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
