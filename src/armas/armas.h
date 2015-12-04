

// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef _ARMAS_H_INCLUDED
#define _ARMAS_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Operarand flag bits
 */
enum armas_flags {
  ARMAS_NOTRANS = 0,
  ARMAS_NULL    = 0,
  ARMAS_NONE    = 0,
  ARMAS_ANY     = 0,
  ARMAS_LOWER   = 0x1,          ///< lower triangular matrix
  ARMAS_UPPER   = 0x2,          ///< upper triangular matrix 
  ARMAS_SYMM    = 0x4,          ///< symmetric matrix
  ARMAS_HERM    = 0x8,          ///< hermitian matrix
  ARMAS_UNIT    = 0x10,         ///< unit diagonal matrix
  ARMAS_LEFT    = 0x20,         ///< multiplicaton from left
  ARMAS_RIGHT   = 0x40,         ///< multiplicaton from right
  ARMAS_TRANSA  = 0x80,         ///< operand A is transposed
  ARMAS_TRANSB  = 0x100,        ///< operand B is transposed
  ARMAS_TRANS   = ARMAS_TRANSA, ///< matrix operand is transposed
  ARMAS_CTRANSA = 0x200,        ///< operand A is conjugate transposed
  ARMAS_CTRANSB = 0x400,        ///< operand B is conjugate transposed
  ARMAS_CTRANS  = ARMAS_CTRANSA,///< matrix operand is conjugate transposed
  ARMAS_MULTQ   = 0x800,        ///< multiply with Q in bidiagonal 
  ARMAS_MULTP   = 0x1000,       ///< multiply with P in bidiagonal 
  ARMAS_WANTQ   = 0x2000,       ///< build the Q matrix in bidiagonal 
  ARMAS_WANTP   = 0x4000,       ///< build the P matrix in bidiagonal 
  ARMAS_WANTU   = 0x8000,       ///< generate left eigenvectors
  ARMAS_WANTV   = 0x10000,      ///< generate right eigenvectors
  ARMAS_FORWARD = 0x20000,      ///< apply forward
  ARMAS_BACKWARD= 0x40000,      ///< apply backward
  ARMAS_ABSA    = 0x80000,      ///< compute with |A| 
  ARMAS_ABSB    = 0x100000,     ///< compute with |B|
  ARMAS_ABS     = ARMAS_ABSA,   ///< compute with |x|
  ARMAS_CONJA   = 0x200000,     ///< conjugate matrix A
  ARMAS_CONJB   = 0x400000,     ///< conjugate matrix B
  ARMAS_CONJ    = ARMAS_CONJA   ///< conjugate
};

enum armas_sort {
  ARMAS_ASC  = 1,       ///< Sort to ascending order
  ARMAS_DESC = -1       ///< Sort to descending order
};

enum armas_opts {
  ARMAS_BLOCKED   = 0,
  ARMAS_SNAIVE    = 0x1,
  ARMAS_KAHAN     = 0x2,
  ARMAS_PAIRWISE  = 0x4,
  ARMAS_RECURSIVE = 0x8,
  ARMAS_BLAS_RECURSIVE = 0x10,  ///< recursive parallel threading
  ARMAS_BLAS_BLOCKED = 0x20,    ///< parallel threading with variable blocksize
  ARMAS_BLAS_TILED = 0x40,      ///< parallel threading with fixed blocksize
  ARMAS_SCHED_ROUNDROBIN = 0x80,///< round-robin scheduling to workers
  ARMAS_SCHED_RANDOM = 0x100,   ///< scheduling to random workers
  ARMAS_SCHED_TWO = 0x200,      ///< scheduling in power-of-two fashion
  ARMAS_BSVD_GOLUB = 0x400,     ///< use Golub algorithm in bidiagonal SVD
  ARMAS_BSVD_DEMMEL = 0x800,    ///< use Demmel-Kahan algorithm in bidiagonal SVD
  ARMAS_ABSTOL = 0x1000,        ///< compute using absolute tolerance
  // renamed versions
  ARMAS_ONAIVE    = 0x1,
  ARMAS_OKAHAN     = 0x2,
  ARMAS_OPAIRWISE  = 0x4,
  ARMAS_ORECURSIVE = 0x8,
  ARMAS_OBLAS_RECURSIVE = 0x10,  ///< recursive parallel threading
  ARMAS_OBLAS_BLOCKED = 0x20,    ///< parallel threading with variable blocksize
  ARMAS_OBLAS_TILED = 0x40,      ///< parallel threading with fixed blocksize
  ARMAS_OSCHED_ROUNDROBIN = 0x80,///< round-robin scheduling to workers
  ARMAS_OSCHED_RANDOM = 0x100,   ///< scheduling to random workers
  ARMAS_OSCHED_TWO = 0x200,      ///< scheduling in power-of-two fashion
  ARMAS_OBSVD_GOLUB = 0x400,     ///< use Golub algorithm in bidiagonal SVD
  ARMAS_OBSVD_DEMMEL = 0x800,    ///< use Demmel-Kahan algorithm in bidiagonal SVD
  ARMAS_OABSTOL = 0x1000,        ///< compute using absolute tolerance
  ARMAS_OEXTPREC = 0x2000        ///< compute using extended precission
};

/**
 * @brief Error codes
 */
enum armas_errors {
  ARMAS_ESIZE        = 1,  ///< operand size mismatch
  ARMAS_ENEED_VECTOR = 2,  ///< vector operand required
  ARMAS_EINVAL = 3,        ///< invalid parameter
  ARMAS_EIMP = 4,          ///< not implemented
  ARMAS_EWORK = 5,         ///< workspace to small
  ARMAS_ESINGULAR = 6,     ///< singular matrix
  ARMAS_ENEGATIVE = 7,     ///< negative value on diagonal
  ARMAS_EMEMORY = 8,       ///< memory allocation failed
  ARMAS_ECONVERGE = 9      ///< algorithm does not converge
};

enum armas_norms {
  ARMAS_NORM_ONE = 1,
  ARMAS_NORM_TWO = 2,
  ARMAS_NORM_INF = 3,
  ARMAS_NORM_FRB = 4
};

#ifndef MAX_CPU
#define MAX_CPU 128
#endif

/**
 * @brief CPU cache-line aligned memory buffer
 */
typedef struct _armas_cbuf_s {
  char *data;           ///< CPU cache-line aligned buffer address
  size_t len;           ///< aligned buffer size
  void *__unaligned;    ///< allocated memory block
  size_t __nbytes;      ///< size of allocated block
  size_t cmem;          ///< requested cache size (L2/L3)
  size_t l1mem;         ///< configured innermost cache size (L1)
} armas_cbuf_t;

/**
 * @brief Configuration parameters
 */
typedef struct armas_conf {
  int mb;               ///< block size relative to result matrix rows (blas3)
  int nb;               ///< block size relative to result matrix cols (blas3)
  int kb;               ///< block size relative to operand matrix common dimension (blas3)
  int lb;               ///< block size for blocked algorithms (lapack)
  int maxproc;          ///< max processors to use
  int wb;               ///< block size for cpu scheduler
  int error;            ///< last error
  int optflags;         ///< config options
  int tolmult;          ///< tolerance multiplier, used tolerance is tolmult*EPSILON
  size_t cmem;          ///< sizeof of internal per-thread cache 
  size_t l1mem;         ///< sizeof of L1 memory
  armas_cbuf_t *cbuf;   ///< user defined cache buffer
} armas_conf_t;

// use default configuration block
#define ARMAS_CDFLT  (armas_conf_t *)0;

extern armas_conf_t *armas_conf_default();
extern long armas_use_nproc(uint64_t nelems, armas_conf_t *conf);
extern int armas_nblocks(uint64_t nelems, int wb, int maxproc, int flags);
extern int armas_last_error();
extern void armas_init(void);

extern armas_cbuf_t *armas_cbuf_default(void);
extern armas_cbuf_t *armas_cbuf_get(armas_conf_t *conf);
extern armas_cbuf_t *armas_cbuf_init(armas_cbuf_t *cbuf, size_t cmem, size_t l1mem);
extern armas_cbuf_t *armas_cbuf_make(armas_cbuf_t *cbuf, void *buf, size_t cmem, size_t l1mem);
extern void armas_cbuf_release(armas_cbuf_t *cbuf);

// pivot vectors

/**
 * @brief Pivot vector
 */
typedef struct armas_pivot {
  int npivots;    ///< Pivot storage size
  int *indexes;   ///< Pivot storage
  int owner;      ///< Storage owner flag
} armas_pivot_t;

#ifndef __INLINE
#define __INLINE extern inline
#endif

/**
 * @brief Initialize pivot vector, allocates new storage.
 */
__INLINE
armas_pivot_t *armas_pivot_init(armas_pivot_t *ptable, int sz)
{
  if (!ptable)
    return ptable;

  ptable->indexes = (int *)calloc(sz, sizeof(int));
  if (!ptable->indexes) {
    return (armas_pivot_t *)0;
  }
  ptable->npivots = sz;
  ptable->owner = 1;
  return ptable;
}

/**
 * @brief Create new pivot vector, allocates new storage.
 */
__INLINE
armas_pivot_t *armas_pivot_new(int sz)
{
  armas_pivot_t *ptable = (armas_pivot_t*)malloc(sizeof(armas_pivot_t));
  if (!ptable)
    return ptable;

  if (! armas_pivot_init(ptable, sz) ) {
    free(ptable);
    return (armas_pivot_t *)0;
  }
  return ptable;
}

/**
 * @brief Setup pivot vector with given storage.
 */
__INLINE
armas_pivot_t *armas_pivot_make(armas_pivot_t *ptable, int sz, int *data)
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
void armas_pivot_release(armas_pivot_t *ptable)
{
  if (ptable) {
    if (ptable->owner)
      free(ptable->indexes);
    ptable->indexes = (int *)0;
    ptable->npivots = 0;
  }
}

/**
 * @brief Free pivot vector.
 */
__INLINE
void armas_pivot_free(armas_pivot_t *ptable)
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
int armas_pivot_size(armas_pivot_t *ptable)
{
  return ptable ? ptable->npivots : 0;
}

/**
 * @brief Get raw pivot storage.
 */
__INLINE
int *armas_pivot_data(armas_pivot_t *ptable)
{
  return ptable ? ptable->indexes : (int *)0;
}

/**
 * @brief Get pivot at index k.
 */
__INLINE
int armas_pivot_get(armas_pivot_t *ptable, int k)
{
  return ptable && k >= 0 && k < ptable->npivots ? ptable->indexes[k] : 0;
}

/**
 * @brief Set pivot at index k.
 */
__INLINE
void armas_pivot_set(armas_pivot_t *ptable, int k, int val)
{
  if (ptable && k >= 0 && k < ptable->npivots)
    ptable->indexes[k] = val;
}

extern void armas_pivot_printf(FILE* out, const char *fmt, armas_pivot_t *pivot);


#ifdef __cplusplus
}
#endif


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
