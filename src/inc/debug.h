
// Copyright (c) Harri Rautila, 2012,2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef __ARMAS_DEBUG_H
#define __ARMAS_DEBUG_H 1

#include <math.h>

#if defined(COMPLEX128)
static inline void __check_nan(double complex val) {}
static inline void __check_mm256(__m256d val) {}

#elif defined(COMPLEX64)
static inline void __check_nan(float complex val) {}
static inline void __check_mm256(__m256d val) {}

#elif defined(FLOAT32)
static inline void __check_nan(float val) {}
static inline void __check_mm256(__m256d val) {}

#else // FLOAT(64)
static inline void __check_nan(double val) {
  double *ptr = 0;
  // if NaN then SIGSEGV 
  if (isnan(val))
    *ptr = val;
}

static inline void __check_mm256(__m256d val) {
  __check_nan(val[0]);
  __check_nan(val[1]);
  __check_nan(val[2]);
  __check_nan(val[3]);
}

#endif

#if defined(__DEBUG__)
#define CHECK_NAN(val)  __check_nan(val)

#define CHECK_MM256(val) __check_mm256(val)

#define IFERROR(exp) do { \
  int _e = (exp); \
  if (_e) { printf("error at: %s:%d\n", __FILE__, __LINE__); } \
  } while (0);

#else
#define CHECK_NAN(val)
#define CHECK_MM256(val)

#define IFERROR(exp) exp

#endif // defined(__DEBUG__)

#endif // __ARMAS_DEBUG_H
  

// Local Variables:
// indent-tabs-mode: nil
// End:
