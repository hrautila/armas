
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_SPARSE_H
#define ARMAS_SPARSE_H 1

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef sp_align64
#define sp_align64(n) (((n)+7) & ~0x7)
#endif
#ifndef sp_nbits_aligned8
#define sp_nbits_aligned8(n) (((n) + 7) >> 3)
#endif

/**
 * @addtogroup sparse
 * @{
 */

typedef ptrdiff_t csint;

typedef enum {
    ARMASSP_UNDEF_ORDER = 0,
    ARMASSP_COL_ORDER = 1,
    ARMASSP_ROW_ORDER = 2
} armassp_order_t;

/**
 * @brief Sparse matrix types.
 */
typedef enum {
    ARMASSP_CSC = 0,   ///< Compressed Sparse Column matrix
    ARMASSP_CSR = 1,   ///< Compressed Sparse Row matrix
    ARMASSP_COO = 2    ///< Coordinate List matrix
} armassp_type_enum;

/** @brief Coordinate list element */
typedef struct coo_elem {
    DTYPE val;    ///< Element value
    int i;        ///< Element row
    int j;        ///< Element column
} coo_elem_t;

/**
 * @brief Sparse matrix
 */
typedef struct armas_sparse {
    union {
        DTYPE *v;
        coo_elem_t *ep;
    } elems;
    int rows;       ///< Matrix row dimension
    int cols;       ///< Matrix columns
    int nnz;        ///< Number of non-zero elements
    int *ptr;       ///< Row/column pointers (size == rows+1 or cols+1) (null for COO)
    int *ix;        ///< Row/column indexes (size == nnz) (null for COO)
    int nptr;       ///< Number of elements in ptr array (== cols || rows)
    int size;       ///< Size of elements buffer
    armassp_type_enum kind;
    size_t __nbytes;
} armas_sparse_t;

/**
 * @brief Sparse column or row of sparse CSC/CSR matrix
 */
typedef struct armas_spvec {
    DTYPE *elems;           ///< Vector elements
    int *ix;                ///< Indexes of non zero elements
    int nz;                 ///< Number of non-zero entries
} armas_spvec_t;

/**
 * @brief Accumulator for sparse-sparse multiply.
 */
typedef struct armas_accum {
    DTYPE *elems;
    int *mark;
    int *queue;
    int nz;
    int tail;
} armas_accum_t;

/**
 * @brief Accumulator memory requirement for matrix of dimension n.
 */
__ARMAS_INLINE
size_t armas_accum_bytes(int n) {
    return __align64(n*sizeof(DTYPE) + 2*n*sizeof(int));
}

/**
 * @brief Get accumulater dimension for matrix.
 */
__ARMAS_INLINE
int armas_accum_dim(const armas_sparse_t *A) {
    return A->kind == ARMASSP_CSR ? A->cols : A->rows;
}

/**
 * @brief Calculate accumulator memory requirement for matrix.
 * @return Number of bytes needed.
 */
__ARMAS_INLINE
size_t armas_accum_need(const armas_sparse_t *A) {
    return armas_accum_bytes(armas_accum_dim(A));
}

extern size_t armas_accum_make(armas_accum_t *acc,
                                 int n, void *ptr, size_t len);
extern int armas_accum_allocate(armas_accum_t *acc, int n);
extern void armas_accum_release(armas_accum_t *acc);
extern void armas_accum_addpos(armas_accum_t *acc,
                                 int k, DTYPE v, int mark);
extern void armas_accum_scatter(armas_accum_t *acc, const armas_spvec_t *x,
                                  DTYPE beta, int mark);
extern void armas_accum_dot(armas_accum_t *acc, int k, const armas_spvec_t *x,
                              const armas_spvec_t *y, int mark);
extern void armas_accum_gather(armas_sparse_t *C, DTYPE alpha,
                                 armas_accum_t *acc, int ik, int maxnz);
extern void armas_accum_clear(armas_accum_t *acc);


/**
 * @brief Sparse matrix preconditioner
 */
typedef struct armassp_precond {
    armas_sparse_t *M;
    int flags;
    int (*precond)(armas_dense_t *z,
                   const struct armassp_precond *M,
                   const armas_dense_t *x, armas_conf_t *cf);
    int (*partial)(armas_dense_t *z,
                   const struct armassp_precond *M,
                   const armas_dense_t *x, int flags, armas_conf_t *cf);
} armassp_precond_t;

extern
int armassp_init_iluz(armassp_precond_t *P, armas_sparse_t *A);
extern
int armassp_init_icholz(armassp_precond_t *P,
                          armas_sparse_t *A, int flags);
extern
void armassp_precond_release(armassp_precond_t *P);

// -------------------------------------------------------------------------
//
/**
 * @brief Calculate memory requirement for sparse matrix.
 *
 * @param[in] rows    Row count
 * @param[in] cols    Columns count
 * @param[in] nnz     Number of non-zero elements
 * @param[in] kind    Sparse matrix type.
 * @return Number of bytes.
 */
__ARMAS_INLINE
size_t armassp_bytes_needed(int rows, int cols,
                              int nnz, armassp_type_enum kind)
{
    if (kind == ARMASSP_COO)
        return nnz*sizeof(coo_elem_t);
    int n = kind == ARMASSP_CSR ? rows : cols;
    return __align64(nnz*sizeof(DTYPE) + sizeof(int)*(nnz + n+1));
}

/**
 * @brief Calculate memory requirement for matrix.
 * @see armassp_bytes_needed
 */
__ARMAS_INLINE
size_t armassp_bytes_for(const armas_sparse_t *A)
{
    return armassp_bytes_needed(A->rows, A->cols, A->nnz, A->kind);
}

__ARMAS_INLINE
size_t armassp_nbytes(const armas_sparse_t *A)
{
    return A->__nbytes;
}

// -------------------------------------------------------------------------
// marking the vertexes on index array


// -------------------------------------------------------------------------
// sparse element access

/**
 * @brief Get start of row/column index in graph.
 */
__ARMAS_INLINE
int armassp_index(const armas_sparse_t *A, int j)
{
    return A->ptr[j];
}

#if 0
// @brief Get safely index position in graph. (never value < 0)
__ARMAS_INLINE
int armassp_index_safe(const armas_sparse_t *A, int j)
{
    return armassp_get_unmarked(A, j);
}
#endif
/**
 * @brief Number of non-zero element on column/row.
 */
__ARMAS_INLINE
int armassp_len(const armas_sparse_t *A, int j)
{
    return A->ptr[j+1]-A->ptr[j];
}
/**
 * @brief Get value of column/row index at position p.
 */
__ARMAS_INLINE
int armassp_at(const armas_sparse_t *A, int p)
{
    return A->ix[p];
}

/**
 * @brief Get pointer to column/row indexes.
 */
__ARMAS_INLINE
int *armassp_iptr(const armas_sparse_t *A, int j)
{
    return &A->ix[A->ptr[j]];
}

/**
 * @brief Get position of element on column/row.
 *
 * @param[in] A   Sparse matrix
 * @param[in] j   Column (CSC) or row (CSR)
 * @param[in] ix  Element index on column/row.
 * @retval >=0  Position non-zero value of element
 * @retval  -1  Element value is zero
 */
__ARMAS_INLINE
int armassp_nz(const armas_sparse_t *A, int j, int ix)
{
    for (int p = A->ptr[j]; A->ix[p] <= ix && p < A->ptr[j+1]; p++) {
        if (A->ix[p] == ix)
            return p;
    }
    return -1;
}
/**
 * @brief Get pointer to non-zero values of column/row.
 */
__ARMAS_INLINE
DTYPE *armassp_data(const armas_sparse_t *A, int j)
{
    return &A->elems.v[armassp_index(A, j)];
}

/**
 * @brief Get non-zero value at position.
 */
__ARMAS_INLINE
DTYPE armassp_value(const armas_sparse_t *A, int p)
{
    return A->elems.v[p];
}

/**
 * @brief Get number of vertexes.
 */
__ARMAS_INLINE
int armassp_nvertex(const armas_sparse_t *A)
{
    return A->nptr;
}

/**
 * @brief Matrix size (number of elements)
 */
__ARMAS_INLINE
int armassp_size(const armas_sparse_t *A)
{
    return A ? A->rows * A->cols : 0;
}

/**
 * @brief Column/row as sparse vector.
 */
__ARMAS_INLINE
armas_spvec_t *armassp_vector(armas_spvec_t *x,
                                  const armas_sparse_t *A, int k)
{
    x->elems = armassp_data(A, k);
    x->ix = armassp_iptr(A, k);
    x->nz = armassp_len(A, k);
    return x;
}

__ARMAS_INLINE
armas_sparse_t *armassp_clear(armas_sparse_t *A) {
    A->ptr = A->ix = (int *)0;
    A->rows = A->cols = A->nnz = A->size = A->nptr = 0;
    A->elems.v = (DTYPE *)0;
    A->__nbytes = 0;
    return A;
}

/**
 * @brief Release sparse matrix resources.
 */
__ARMAS_INLINE
void armassp_release(armas_sparse_t *A) {
    if (A->elems.v)
        free(A->elems.v);
    armassp_clear(A);
}

/**
 * @brief Free sparse matrix and its resources.
 */
__ARMAS_INLINE
void armassp_free(armas_sparse_t *A)
{
    if (A) {
        armassp_release(A);
        free(A);
    }
}


extern
int armassp_make(armas_sparse_t *A, int rows, int cols, int nnz,
                   armassp_type_enum storage, void *data, size_t dlen);
extern
armas_sparse_t *armassp_init(armas_sparse_t *A, int rows,
                                 int cols, int nnz, armassp_type_enum kind);
extern
armas_sparse_t *armassp_new(int rows, int cols, int nnz,
                                armassp_type_enum kind);
extern
armas_sparse_t *armassp_spa_alloc(armas_sparse_t *A, int nnz);
extern
int armassp_append(armas_sparse_t *A, int row,
                     int col, DTYPE val);
extern
int armassp_resize(armas_sparse_t *A, int newsize);
extern
int armassp_sort_to(armas_sparse_t *A, armassp_order_t order);
extern
int armassp_sort(armas_sparse_t *A);

extern
armas_sparse_t *armassp_convert(const armas_sparse_t *B,
                                    armassp_type_enum target);
extern
armas_sparse_t *armassp_convert_to(armas_sparse_t *A,
                                       const armas_sparse_t *B,
                                       armassp_type_enum target);
extern
armas_sparse_t *armassp_transpose(const armas_sparse_t *B);
extern
armas_sparse_t *armassp_transpose_to(armas_sparse_t *A,
                                         const armas_sparse_t *B);
extern
armas_sparse_t *armassp_copy_to(armas_sparse_t *A,
                                    const armas_sparse_t *B);
extern
armas_sparse_t *armassp_mkcopy(const armas_sparse_t *B);

extern
armas_sparse_t *armassp_mmload(int *typecode, FILE *f);
extern
int armassp_mmdump(FILE *f, const armas_sparse_t *A, int flags);
extern
void armassp_pprintf(FILE *f, const armas_sparse_t *A);
extern
void armassp_iprintf(FILE *f, const armas_sparse_t *A);
extern
int armassp_todense(armas_dense_t *A,
                      const armas_sparse_t *B, armas_conf_t *cf);

extern
int armassp_mvmult_trm(armas_dense_t *x, DTYPE alpha,
                         const armas_sparse_t *A,
                         int flags, armas_conf_t *cf);
extern
int armassp_mvsolve_trm(armas_dense_t *x, DTYPE alpha,
                          const armas_sparse_t *A,
                          int flags, armas_conf_t *cf);
extern
int armassp_mvmult(DTYPE beta, armas_dense_t *y,
                     DTYPE alpha, const armas_sparse_t *A,
                     const armas_dense_t *x, int flags,
                     armas_conf_t *cf);
extern
int armassp_mvmult_sym(DTYPE beta, armas_dense_t *y,
                         DTYPE alpha, const armas_sparse_t *A,
                         const armas_dense_t *x, int flags,
                         armas_conf_t *cf);

extern
int armassp_cgrad(armas_dense_t *x, const armas_sparse_t *A,
                    const armas_dense_t *b,int flags, armas_conf_t *cf);
extern
int armassp_cgrad_w(armas_dense_t *x, const armas_sparse_t *A,
                      const armas_dense_t *b, int flags, armas_wbuf_t *W,
                      armas_conf_t *cf);
extern
int armassp_pcgrad_w(armas_dense_t *x, const armas_sparse_t *A,
                       const armas_dense_t *b, armassp_precond_t *P,
                       int flags, armas_wbuf_t *W, armas_conf_t *cf);
extern
int armassp_pcgrad(armas_dense_t *x, const armas_sparse_t *A,
                     const armas_dense_t *b, armassp_precond_t *P,
                     int flags, armas_conf_t *cf);

extern
int armassp_cgnr(armas_dense_t *x, const armas_sparse_t *A,
                   const armas_dense_t *b,armas_conf_t *cf);
extern
int armassp_cgnr_w(armas_dense_t *x, const armas_sparse_t *A,
                     const armas_dense_t *b, armas_wbuf_t *W,
                     armas_conf_t *cf);
extern
int armassp_cgne(armas_dense_t *x, const armas_sparse_t *A,
                   const armas_dense_t *b, armas_conf_t *cf);
extern
int armassp_cgne_w(armas_dense_t *x, const armas_sparse_t *A,
                     const armas_dense_t *b,armas_wbuf_t *W,
                     armas_conf_t *cf);
extern
int armassp_gmres(armas_dense_t *x, const armas_sparse_t *A,
                    const armas_dense_t *b,armas_conf_t *cf);
extern
int armassp_gmres_w(armas_dense_t *x,  const armas_sparse_t *A,
                      const armas_dense_t *b, armas_wbuf_t *W,
                      armas_conf_t *cf);
extern
int armassp_pgmres(armas_dense_t *x, const armas_sparse_t *A,
                     const armas_dense_t *b,const armassp_precond_t *M,
                     armas_conf_t *cf);
extern
int armassp_pgmres_w(armas_dense_t *x,  const armas_sparse_t *A,
                       const armas_dense_t *b, const armassp_precond_t *M,
                       armas_wbuf_t *W, armas_conf_t *cf);

extern
int armassp_addto_w(armas_sparse_t *C, DTYPE alpha, const armas_sparse_t *A,
                      DTYPE beta, const armas_sparse_t *B, int bits,
                      armas_wbuf_t *work,armas_conf_t *cf);
extern
armas_sparse_t *armassp_add(DTYPE alpha, const armas_sparse_t *A,
                                DTYPE beta, const armas_sparse_t *B,
                                int bits, armas_conf_t *cf);

extern
int armassp_multto_w(armas_sparse_t *C, DTYPE alpha,
                       const armas_sparse_t *A, const armas_sparse_t *B,
                       int bits, armas_wbuf_t *work, armas_conf_t *cf);
extern
armas_sparse_t *armassp_mult(DTYPE alpha, const armas_sparse_t *A,
                                 const armas_sparse_t *B, int bits,
                                 armas_conf_t *cf);

extern
int armassp_init_icholz(armassp_precond_t *P,
                          armas_sparse_t *A, int flags);
extern
int armassp_icholz(armas_sparse_t *A, int flags);
extern
int armassp_iluz(armas_sparse_t *L);

extern int armassp_hasdiag(const armas_sparse_t *A, int diag);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* ARMASSP_SPARSE */
