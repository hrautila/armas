
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_COND_H 
#define __ARMAS_COND_H 1

#if defined(HAVE_CONFIG_H)
  #include <config.h>
#endif

// Here are macros for conditional compilation code blocks. This file should be
// included just before actual code when all per file macros are defined. (see blas3/gemm.c)

// EXT_PRECISION is global define; HAVE_EXT_PRECISION is per file define
// that is true if extended precision implementation is available in current compilation
#if defined(EXT_PRECISION) && defined(HAVE_EXT_PRECISION)

// If <flag> expression TRUE, call <func> and return with it's return value
#define IF_EXTPREC(flag, func)			\
    do {                                        \
        if (flag) {                             \
            return func;                        \
        }                                       \
    } while(0)

// If <flag> expression TRUE, call <func> and return with specified rval
#define IF_EXTPREC_RVAL(flag, rval, func)	\
    do {                                        \
        if (flag) {                             \
            func; return rval;			\
        }                                       \
    } while(0)

#else

#define IF_EXTPREC(flag, func) 

#define IF_EXTPREC_RVAL(flag, rval, func) 

#endif // EXT_PRECISION

#endif  // __ARMAS_COND_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:


