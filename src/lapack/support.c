
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#define __ARMAS_PROVIDES 1
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

// calculate maximum blocking size for an operation. 
int compute_lb(int M, int N, int wsz, WSSIZE worksize)
{
  int lb = 0;
  int k = 0;
  int wblk = (*worksize)(M, N, 4);
  if (wsz < wblk) {
    return 0;
  }
  do {
    lb += 4;
    wblk = (*worksize)(M, N, lb+4);
    k++;
  } while  (wsz > wblk && k < 100);
  if (k == 100)
    return 0;

  if (wblk > wsz) {
    lb -= 4;
  }
  return lb < 0 ? 0 : lb;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
