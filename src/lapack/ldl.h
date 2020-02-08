
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_LAPACK_LDL_H
#define ARMAS_LAPACK_LDL_H

#include "sym.h"

// --------------------------------------------------------------------------------
// type dependent names for internal ldl-related functions.
#if defined(COMPLEX64)
// single precision complex

#elif defined(COMPLEX128)
// double precision complex

#elif defined(FLOAT32)
// single precision real
#define ldlfactor_np   s_ldlfactor_np
#define ldlsolve_np    s_ldlsolve_np

#else
// double precision
#define ldlfactor_np   d_ldlfactor_np
#define ldlsolve_np    d_ldlsolve_np
#endif
// --------------------------------------------------------------------------------

static inline
int ws_calculate(int rows, int lb)
{
    return lb > 0 ? (rows - lb) * lb : 0;
}

static inline
int new_lb(int rows, int lb, int ws)
{
    if (lb > 0 && lb * (rows - lb) < ws)
        return lb;
    // worksize not big enough for this blocking size (lb)
    // solve: 
    //    lb*(rows-lb) - ws = 0 ==> -lb^2 + rows*lb - ws = 0 ==>
    //    lb^2 - rows*lb + ws = 0 ==>
    //    lb = 0.5*(rows - sqrt(rows^2 - 4*w))
    double r = (double) rows, w = (double) ws;
    lb = (int) (0.5 * (r - SQRT(r * r - 4 * w)));
    // round down to closest multiple of four
    lb -= (lb % 4);
    return lb;
}

#endif // ARMAS_LAPACK_LDL_H
