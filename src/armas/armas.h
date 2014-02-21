

// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef _ARMAS_H_INCLUDED
#define _ARMAS_H_INCLUDED

#include <stdint.h>
#include <stdlib.h>

/**
 * @brief Operarand flag bits
 */
enum armas_flags {
  ARMAS_NOTRANS = 0,
  ARMAS_NULL    = 0,
  ARMAS_NONE    = 0,
  ARMAS_TRANSA  = 0x1,     ///< operand A is transposed
  ARMAS_TRANSB  = 0x2,     ///< operand B is transposed
  ARMAS_TRANS   = 0x4,     ///< matrix operand is transposed
  ARMAS_LOWER   = 0x8,     ///< lower triangular matrix
  ARMAS_UPPER   = 0x10,    ///< upper triangular matrix 
  ARMAS_LEFT    = 0x20,    ///< multiplicaton from left
  ARMAS_RIGHT   = 0x40,    ///< multiplicaton from right
  ARMAS_UNIT    = 0x80,    ///< unit diagonal matrix
  ARMAS_CONJA   = 0x100,   ///< operand A is conjugate transposed
  ARMAS_CONJB   = 0x200,   ///< operand B is conjugate transposed
  ARMAS_CONJ    = 0x400,   ///< matrix operand is conjugate transposed
  ARMAS_SYMM    = 0x800,   ///< symmetric matrix
  ARMAS_HERM    = 0x1000   ///< hermitian matrix
};

enum armas_opts {
  ARMAS_BLOCKED   = 0,
  ARMAS_SNAIVE    = 0x1,
  ARMAS_KAHAN     = 0x2,
  ARMAS_PAIRWISE  = 0x4,
  ARMAS_RECURSIVE = 0x8
};

/**
 * @brief Error codes
 */
enum armas_errors {
  ARMAS_ESIZE        = 1,  ///< operand size mismatch
  ARMAS_ENEED_VECTOR = 2,  ///< vector operand required
  ARMAS_EINVAL = 3,        ///< invalid parameter
  ARMAS_EIMP = 4           ///< not implemented
};

enum armas_norms {
  ARMAS_NORM_ONE = 1,
  ARMAS_NORM_TWO = 2,
  ARMAS_NORM_INF = 3
};

#ifndef MAX_CPU
#define MAX_CPU 128
#endif

/**
 * @brief Configuration parameters
 */
typedef struct armas_conf {
  int mb;        ///< block size relative to result matrix rows
  int nb;        ///< block size relative to result matrix cols
  int kb;        ///< block size relative to operand matrix common dimension 
  int lb;        ///< block size for blocked algorithms 
  int maxproc;   ///< max processors to use
  int error;     ///< last error
  int optflags;  ///< config options
} armas_conf_t;

// use default configuration block
#define ARMAS_CDFLT  (armas_conf_t *)0;

extern void armas_nproc_schedule(uint64_t *nproc_schedule, int count);

extern long armas_use_nproc(uint64_t nelems, armas_conf_t *conf);

extern armas_conf_t *armas_conf_default();

extern int armas_last_error();

// pivot vectors

/**
 * @brief Pivot vector
 */
typedef struct armas_pivots {
  int npivots;    ///< Pivot storage size
  int *indexes;   ///< Pivot storage
  int owner;      ///< Storage owner flag
} armas_pivots_t;

#ifndef __INLINE
#define __INLINE extern inline
#endif

/**
 * @brief Create new pivot vector, allocates new storage.
 */
__INLINE
armas_pivots_t *armas_pivot_new(int sz)
{
  armas_pivots_t *ptable = (armas_pivots_t*)malloc(sizeof(armas_pivots_t));
  if (!ptable)
    return ptable;

  ptable->indexes = (int *)calloc(sz, sizeof(int));
  if (!ptable->indexes) {
    free(ptable);
    return (armas_pivots_t *)0;
  }
  ptable->npivots = sz;
  ptable->owner = 1;
  return ptable;
}

/**
 * @brief Setup pivot vector with given storage.
 */
__INLINE
armas_pivots_t *armas_pivot_make(armas_pivots_t *ptable, int sz, int *data)
{
  ptable->npivots = sz;
  ptable->indexes = data;
  ptable->owner = 0;
  return ptable;
}

/**
 * @brief Release pivot vector storage.
 */
__INLINE
void armas_pivot_release(armas_pivots_t *ptable)
{
  if (ptable && ptable->owner) {
    free(ptable->indexes);
    ptable->indexes = (int *)0;
  }
}

/**
 * @brief Free pivot vector.
 */
__INLINE
void armas_pivot_free(armas_pivots_t *ptable)
{
  if (!ptable)
    return;
  if (ptable->owner)
    free(ptable->indexes);
  free(ptable);
}

/**
 * @brief Pivot vector size.
 */
__INLINE
int armas_pivot_size(armas_pivots_t *ptable)
{
  return ptable ? ptable->npivots : 0;
}

/**
 * @brief Get raw pivot storage.
 */
__INLINE
int *armas_pivot_data(armas_pivots_t *ptable)
{
  return ptable ? ptable->indexes : (int *)0;
}




#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
