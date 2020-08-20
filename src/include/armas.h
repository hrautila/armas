
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

/**
 * \file
 * Type independent defintions.
 */

#ifndef _ARMAS_H_INCLUDED
#define _ARMAS_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __ARMAS_INLINE
    #if __GNUC__
        #if !__STDC_VERSION__
            #define __ARMAS_INLINE extern inline
        #else
            #define __ARMAS_INLINE inline
        #endif
    #else
        #define __ARMAS_INLINE inline
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Libtool version numbering
 */
#define ARMAS_ABI_CURRENT 1
#define ARMAS_ABI_REVISION 0
#define ARMAS_ABI_AGE 0

/**
 * @brief Operarand flag bits
 */
enum armas_flags {
    ARMAS_NOTRANS = 0,
    ARMAS_NULL = 0,
    ARMAS_NONE = 0,
    ARMAS_ANY = 0,
    ARMAS_LOWER = 0x1,             ///< lower triangular matrix
    ARMAS_UPPER = 0x2,             ///< upper triangular matrix
    ARMAS_SYMM = 0x4,              ///< symmetric matrix
    ARMAS_HERM = 0x8,              ///< hermitian matrix
    ARMAS_UNIT = 0x10,             ///< unit diagonal matrix
    ARMAS_LEFT = 0x20,             ///< multiplicaton from left
    ARMAS_RIGHT = 0x40,            ///< multiplicaton from right
    ARMAS_TRANSA = 0x80,           ///< operand A is transposed
    ARMAS_TRANSB = 0x100,          ///< operand B is transposed
    ARMAS_TRANS = ARMAS_TRANSA,    ///< matrix operand is transposed
    ARMAS_CTRANSA = 0x200,         ///< operand A is conjugate transposed
    ARMAS_CTRANSB = 0x400,         ///< operand B is conjugate transposed
    ARMAS_CTRANS = ARMAS_CTRANSA,  ///< matrix operand is conjugate transposed
    ARMAS_MULTQ = 0x800,           ///< multiply with Q in bidiagonal
    ARMAS_MULTP = 0x1000,          ///< multiply with P in bidiagonal
    ARMAS_WANTQ = 0x2000,          ///< build the Q matrix in bidiagonal
    ARMAS_WANTP = 0x4000,          ///< build the P matrix in bidiagonal
    ARMAS_WANTU = 0x8000,          ///< generate left eigenvectors
    ARMAS_WANTV = 0x10000,         ///< generate right eigenvectors
    ARMAS_FORWARD = 0x20000,       ///< apply forward
    ARMAS_BACKWARD = 0x40000,      ///< apply backward
    ARMAS_ABSA = 0x80000,          ///< compute with |A|
    ARMAS_ABSB = 0x100000,         ///< compute with |B|
    ARMAS_ABS = ARMAS_ABSA,        ///< compute with |x|
    ARMAS_CONJA = 0x200000,        ///< conjugate matrix A
    ARMAS_CONJB = 0x400000,        ///< conjugate matrix B
    ARMAS_CONJ = ARMAS_CONJA,      ///< conjugate
    ARMAS_HHNEGATIVE = 0x800000,   ///< compute Householder for [-beta; 0]
    ARMAS_NONNEG = 0x1000000       ///< request non-negative result (householder)
};

/**
 * @brief Sorting orders.
 */
enum armas_sort {
    ARMAS_ASC = 1,   ///< Sort to ascending order
    ARMAS_DESC = -1  ///< Sort to descending order
};

/**
 * @brief Pivot directions.
 */
enum armas_pivots {
    ARMAS_PIVOT_FORWARD = 0x0,   ///< Pivots forwards
    ARMAS_PIVOT_BACKWARD = 0x1,  ///< Pivot backwards
    ARMAS_PIVOT_ROWS = 0x2,      ///< Pivot rows
    ARMAS_PIVOT_COLS = 0x4,      ///< Pivot columns
    ARMAS_PIVOT_UPPER = 0x8,     ///< Pivot upper triangular symmetric matrix
    ARMAS_PIVOT_LOWER = 0x10     ///< Pivot lower triangular symmetric matrix
};

/**
 * @brief Configuration options
 */
enum armas_opts {
    // renamed versions
    ARMAS_ONAIVE = 0x1,
    ARMAS_OKAHAN = 0x2,
    ARMAS_OPAIRWISE = 0x4,
    ARMAS_ORECURSIVE = 0x8,
    ARMAS_OBLAS_RECURSIVE = 0x10,    ///< recursive parallel threading
    ARMAS_OBLAS_BLOCKED = 0x20,      ///< parallel threading with variable blocksize
    ARMAS_OBLAS_TILED = 0x40,        ///< parallel threading with fixed blocksize
    ARMAS_OSCHED_ROUNDROBIN = 0x80,  ///< round-robin scheduling to workers
    ARMAS_OSCHED_RANDOM = 0x100,     ///< scheduling to random workers
    ARMAS_OSCHED_TWO = 0x200,        ///< scheduling in power-of-two fashion
    ARMAS_OBSVD_GOLUB = 0x400,       ///< use Golub algorithm in bidiagonal SVD
    ARMAS_OBSVD_DEMMEL = 0x800,      ///< use Demmel-Kahan algorithm in bidiagonal SVD
    ARMAS_OABSTOL = 0x1000,          ///< compute using absolute tolerance
    ARMAS_OEXTPREC = 0x2000,         ///< compute using extended precission
    ARMAS_ONONNEG = 0x4000,          ///< compute Householder with non-negative diagonal
    ARMAS_CBUF_THREAD = 0x8000,      ///<
    ARMAS_CBUF_LOCAL = 0x10000
};

/**
 * @brief Error codes
 */
enum armas_errors {
    ARMAS_ESIZE = 1,         ///< operand size mismatch
    ARMAS_ENEED_VECTOR = 2,  ///< vector operand required
    ARMAS_EINVAL = 3,        ///< invalid parameter
    ARMAS_EIMP = 4,          ///< not implemented
    ARMAS_EWORK = 5,         ///< workspace too small
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

/**
 * @brief Matrix I/O bits
 */
enum armas_mmbits {
    ARMAS_MM_HEADER = 0x1,
    ARMAS_MM_MATRIX = 0x2,
    ARMAS_MM_REAL = 0x4,
    ARMAS_MM_COMPLEX = 0x8,
    ARMAS_MM_INTEGER = 0x10,
    ARMAS_MM_ARRAY = 0x20,
    ARMAS_MM_COORDINATE = 0x40,
    ARMAS_MM_SYMMETRIC = 0x80,
    ARMAS_MM_HERMITIAN = 0x100,
    ARMAS_MM_GENERAL = 0x200
};

/**
   * @brief JSON parser tokens excluding simple tokens '{', '}', '[', ']', ':', ','
   */
enum armas_json_tokens {
    ARMAS_JSON_STRING = 0x7fffff,
    ARMAS_JSON_INT,
    ARMAS_JSON_NUMBER,
    ARMAS_JSON_TRUE,
    ARMAS_JSON_FALSE,
    ARMAS_JSON_NULL,
    ARMAS_JSON_EINVAL,
    ARMAS_JSON_E2BIG,
    ARMAS_JSON_EOF
};

#ifdef ARMAS_WITH_CHECKS
    // with caveats about NDEBUG (see assert() man page)
    #include <assert.h>
    #define require(x) assert(x)
#else
    #define require(x)
#endif /* ARMAS_WITH_CHECKS */

extern const char *ARMAS_AC_THREADED;
extern const char *ARMAS_AC_WORKERS;

/**
 * @brief Library configuration block.
 */
typedef struct armas_env {
    int mb;        ///< block size relative to result matrix rows (blas3)
    int nb;        ///< block size relative to result matrix cols (blas3)
    int kb;        ///< block size relative to operand matrix common dimension (blas3)
    int lb;        ///< block size for blocked algorithms (lapack)
    int blas1min;  ///< Blas1 functions minimum length for recursive implementation.
    int blas2min;  ///< Blas2: minimum size of dimension to use recursive implementation.
    size_t cmem;   ///< sizeof of internal per-thread cache
    size_t l1mem;  ///< sizeof of L1 memory
    int fixed;     ///< if non-zero, then mb,nb,kb set to fixed values.
} armas_env_t;

/// @brief Get global blocking configuration
armas_env_t *armas_getenv();

/**
 * @brief CPU cache-line aligned memory buffer
 */
typedef struct armas_cbuf {
    char *data;         ///< CPU cache-line aligned buffer address
    size_t len;         ///< aligned buffer size
    void *__unaligned;  ///< allocated memory block
    size_t __nbytes;    ///< size of allocated block
    size_t cmem;        ///< requested cache size (L2/L3)
    size_t l1mem;       ///< configured innermost cache size (L1)
} armas_cbuf_t;

#define ARMAS_CBUF_EMPTY \
    (armas_cbuf_t) { .data = (char *)0, .__unaligned = (void *)0, .__nbytes = 0, .len = 0 }

struct armas_wbuf;
/**
 * @brief Opaque handle to accelerator object.
 */
typedef void *armas_ac_handle_t;

/**
 * @brief Configuration parameters
 */
typedef struct armas_conf {
    int error;                ///< last error
    int optflags;             ///< config options
    int tolmult;              ///< tolerance multiplier, used tolerance is tolmult*EPSILON
    struct armas_wbuf *work;  ///< user defined space for cache buffer
    armas_ac_handle_t accel;
    // -- parameters for iterative methods
    int maxiter;      ///< Max iterations allowed
    int gmres_m;      ///< Number of columns in GMRES
    int numiters;     ///< Number of iterations used (output)
    double stop;      ///< Absolute stopping criterion
    double smult;     ///< Relative stopping criterion multiplier
    double residual;  ///< Result error residual (output)
} armas_conf_t;

// use default configuration block
#define ARMAS_CDFLT (armas_conf_t *)0;

extern const char *armas_version();
extern const char *armas_name();
extern const char **armas_config_options();
extern armas_conf_t *armas_conf_default();
extern int armas_last_error();
extern void armas_init(void);

extern int armas_ac_init(armas_ac_handle_t *handle, const char *name);
extern void armas_ac_release(armas_ac_handle_t handle);
extern int armas_ac_dispatch(
    armas_ac_handle_t handle, int opcode, void *args, struct armas_conf *cf);

extern armas_cbuf_t *armas_cbuf_default(void);
extern armas_cbuf_t *armas_cbuf_get(armas_conf_t *conf);
extern armas_cbuf_t *armas_cbuf_init(armas_cbuf_t *cbuf, size_t cmem, size_t l1mem);
extern armas_cbuf_t *armas_cbuf_make(armas_cbuf_t *cbuf, void *buf, size_t cmem, size_t l1mem);
extern void armas_cbuf_release(armas_cbuf_t *cbuf);
extern armas_cbuf_t *armas_cbuf_create_thread_global();
extern armas_cbuf_t *armas_cbuf_get_thread_global();
extern void armas_cbuf_release_thread_global();
extern int armas_cbuf_select(armas_cbuf_t *cbuf, armas_conf_t *cf);

/**
 * @brief I/O stream functions
 */
typedef struct armas_iostream_vtable {
    int (*get_char)(void *stream);
    void (*unget_char)(void *stream, int c);
    int (*put_char)(void *stream, int c);
} armas_iostream_vtable_t;

/**
 * @brief Very simple I/O stream
 */
typedef struct armas_iostream {
    void *uptr;
    armas_iostream_vtable_t *vt;
} armas_iostream_t;

/**
 * @brief Initialize I/O stream
 */
__ARMAS_INLINE
void armas_iostream_init(armas_iostream_t *iostream, armas_iostream_vtable_t *vt, void *ptr)
{
    if (iostream) {
        iostream->uptr = ptr;
        iostream->vt = vt;
    }
}

/**
 * @brief Get character from input stream
 */
__ARMAS_INLINE
int armas_getchar(armas_iostream_t *stream)
{
    if (!stream)
        return -1;
    return stream->vt->get_char(stream->uptr);
}

/**
 * @brief Put character back to input stream
 */
__ARMAS_INLINE
void armas_ungetchar(armas_iostream_t *stream, int c)
{
    if (!stream)
        return;
    stream->vt->unget_char(stream->uptr, c);
}

/**
 * @brief Put character to output stream
 */
__ARMAS_INLINE
int armas_putchar(armas_iostream_t *stream, int c)
{
    if (!stream)
        return -1;
    return stream->vt->put_char(stream->uptr, c);
}

extern int armas_json_read_token(char *iobuf, size_t len, armas_iostream_t *ios);
extern int armas_json_write_token(int tok, const void *ptr, size_t len, armas_iostream_t *ios);
extern int armas_json_write_simple_token(int tok, armas_iostream_t *ios);

// pivot vectors

/**
 * @brief Pivot vector
 */
typedef struct armas_pivot {
    int npivots;   ///< Pivot storage size
    int *indexes;  ///< Pivot storage
    int owner;     ///< Storage owner flag
} armas_pivot_t;

///< No pivoting indicator
#define ARMAS_NOPIVOT (armas_pivot_t *)0

/**
 * @brief Initialize pivot vector, allocates new storage.
 */
__ARMAS_INLINE
armas_pivot_t *armas_pivot_init(armas_pivot_t *ptable, int sz)
{
    if (!ptable)
        return ptable;

    if (sz == 0) {
        ptable->indexes = (int *)0;
        ptable->npivots = 0;
        ptable->owner = 0;
        return ptable;
    }

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
__ARMAS_INLINE
armas_pivot_t *armas_pivot_new(int sz)
{
    armas_pivot_t *ptable = (armas_pivot_t *)malloc(sizeof(armas_pivot_t));
    if (!ptable)
        return ptable;

    if (!armas_pivot_init(ptable, sz)) {
        free(ptable);
        return (armas_pivot_t *)0;
    }
    return ptable;
}

/**
 * @brief Setup pivot vector with given storage.
 */
__ARMAS_INLINE
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
__ARMAS_INLINE
void armas_pivot_release(armas_pivot_t *ptable)
{
    if (ptable) {
        if (ptable->owner)
            free(ptable->indexes);
        ptable->indexes = (int *)0;
        ptable->npivots = 0;
        ptable->owner = 0;
    }
}

/**
 * @brief Free pivot vector.
 */
__ARMAS_INLINE
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
__ARMAS_INLINE
int armas_pivot_size(const armas_pivot_t *ptable)
{
    return ptable ? ptable->npivots : 0;
}

/**
 * @brief Get raw pivot storage.
 */
__ARMAS_INLINE
int *armas_pivot_data(armas_pivot_t *ptable)
{
    return ptable ? ptable->indexes : (int *)0;
}

/**
 * @brief Get pivot at index k.
 */
__ARMAS_INLINE
int armas_pivot_get(const armas_pivot_t *ptable, int k)
{
    return ptable && k >= 0 && k < ptable->npivots ? ptable->indexes[k] : 0;
}

__ARMAS_INLINE
int armas_pivot_get_unsafe(const armas_pivot_t *ptable, int k)
{
    return ptable->indexes[k];
}

/**
 * @brief Set pivot at index k.
 */
__ARMAS_INLINE
void armas_pivot_set(armas_pivot_t *ptable, int k, int val)
{
    if (ptable && k >= 0 && k < ptable->npivots)
        ptable->indexes[k] = val;
}

__ARMAS_INLINE
void armas_pivot_set_unsafe(armas_pivot_t *ptable, int k, int val)
{
    ptable->indexes[k] = val;
}

extern void armas_pivot_printf(FILE *out, const char *fmt, const armas_pivot_t *pivot);

// ------------------------------------------------------------------------------
// workspace

/**
 * \brief Workspace buffer
 */
typedef struct armas_wbuf {
    char *buf;
    size_t bytes;
    size_t offset;
} armas_wbuf_t;

#define ARMAS_WBNULL \
    (armas_wbuf_t) { .buf = (char *)0, .bytes = 0, .offset = 0 }
#define ARMAS_NOWORK (armas_wbuf_t *)0

#define __align64(n) (((n) + 7) & ~0x7)
#define __nbits_aligned8(n) (((n) + 7) >> 3)

// \brief Allocate nbytes of workspace
__ARMAS_INLINE
armas_wbuf_t *armas_walloc(armas_wbuf_t *W, size_t nbytes)
{
    W->buf = (char *)calloc(nbytes, 1);
    W->bytes = W->buf ? nbytes : 0;
    W->offset = 0;
    return W->buf ? W : (armas_wbuf_t *)0;
}
// \brief Release workspace allocation
__ARMAS_INLINE
void armas_wrelease(armas_wbuf_t *W)
{
    if (W && W->buf) {
        free(W->buf);
        W->buf = (char *)0;
        W->bytes = W->offset = 0;
    }
}

// \brief Reserve nbytes from workspace (aligned to 64bit access)
__ARMAS_INLINE
void *armas_wreserve_bytes(armas_wbuf_t *W, size_t nbytes)
{
    nbytes = __align64(nbytes);
    if (!W || nbytes > W->bytes - W->offset)
        return (char *)0;
    char *r = &W->buf[W->offset];
    W->offset += nbytes;
    return (void *)r;
}

// \brief Reserve space for count number of elements of size 'sz'
__ARMAS_INLINE
void *armas_wreserve(armas_wbuf_t *W, size_t count, size_t sz)
{
    return armas_wreserve_bytes(W, count * sz);
}

__ARMAS_INLINE
void *armas_wreserve_bits(armas_wbuf_t *W, size_t count)
{
    size_t bc = __nbits_aligned8(count);
    return armas_wreserve_bytes(W, bc);
}

// \brief Reset work space reservations
__ARMAS_INLINE
void armas_wreset(armas_wbuf_t *W)
{
    W->offset = 0;
}

// \brief Zero workspace, at most n bytes or all (n == 0)
__ARMAS_INLINE
void armas_wzero(armas_wbuf_t *W, size_t n)
{
    memset(W->buf, 0, n > 0 && n <= W->bytes ? n : W->bytes);
}

// \brief Get size of unreserved space
__ARMAS_INLINE
size_t armas_wbytes(const armas_wbuf_t *W)
{
    return W ? W->bytes - W->offset : 0;
}

// \brief Get current offset
__ARMAS_INLINE
size_t armas_wpos(const armas_wbuf_t *W)
{
    return W ? W->offset : 0;
}

// \brief Get current pointer
__ARMAS_INLINE
void *armas_wptr(const armas_wbuf_t *W)
{
    return W ? &W->buf[W->offset] : (void *)0;
}

// \brief Set current offset to defined value
__ARMAS_INLINE
void armas_wsetpos(armas_wbuf_t *W, size_t pos)
{
    if (W && pos < W->bytes)
        W->offset = pos;
}
#ifdef __cplusplus
}
#endif

#endif /* ARMAS_H_INCLUDED */
