

// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef _ARMAS_H_INCLUDED
#define _ARMAS_H_INCLUDED

#include <stdint.h>

enum armas_flags {
  ARMAS_NOTRANS = 0,
  ARMAS_NULL    = 0,
  // operand A is transposed
  ARMAS_TRANSA  = 0x1,          
  // operand B is transposed
  ARMAS_TRANSB  = 0x2,
  // matrix operand is transposed
  ARMAS_TRANS   = 0x4,
  // lower triangular matrix
  ARMAS_LOWER   = 0x8,
  // upper triangular matrix 
  ARMAS_UPPER   = 0x10,
  // multiplicaton from left
  ARMAS_LEFT    = 0x20,
  // multiplicaton from right
  ARMAS_RIGHT   = 0x40,
  // unit diagonal matrix
  ARMAS_UNIT    = 0x100,
  // symmetric matrix
  ARMAS_SYMM    = 0x200,
  // operand A is conjugate transposed
  ARMAS_CONJA   = 0x400,
  // operand B is conjugate transposed
  ARMAS_CONJB   = 0x1000,
  // matrix operand is conjugate transposed
  ARMAS_CONJ    = 0x2000,
};

enum armas_opts {
  ARMAS_BLOCKED   = 0,
  ARMAS_SNAIVE    = 0x1,
  ARMAS_KAHAN     = 0x2,
  ARMAS_PAIRWISE  = 0x4,
  ARMAS_RECURSIVE = 0x8
};

enum armas_errors {
  // operand size mismatch
  ARMAS_ESIZE        = 1,
  // vector operand required
  ARMAS_ENEED_VECTOR = 2,
  // invalid parameter
  ARMAS_EINVAL = 3,
  // not implemented
  ARMAS_EIMP = 4
};

enum armas_norms {
  ARMAS_NORM_ONE = 1,
  ARMAS_NORM_TWO = 2,
  ARMAS_NORM_INF = 3
};

#ifndef MAX_CPU
#define MAX_CPU 128
#endif

typedef struct armas_conf {
  // block size relative to result matrix rows
  int mb;
  // block size relative to result matrix cols
  int nb;
  // block size relative to operand matrix common dimension
  int kb;       
  // block size for blocked algorithms
  int wb;
  // max processors to use
  int maxproc;  
  // last error
  int error;    
  // config options
  int optflags;

} armas_conf_t;

// use default configuration block
#define ARMAS_CDFLT  (armas_conf_t *)0;

extern void armas_nproc_schedule(uint64_t *nproc_schedule, int count);

extern long armas_use_nproc(uint64_t nelems, armas_conf_t *conf);

extern armas_conf_t *armas_conf_default();

extern int armas_last_error();

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
