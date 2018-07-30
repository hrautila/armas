// Copyright (c) Harri Rautila, 2017

#ifndef __ARMAS_SPARSE_H
#define __ARMAS_SPARSE_H 1

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include <armas/armas.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __align64
#define __align64(n) (((n)+7) & ~0x7)
#endif
#ifndef __nbits_aligned8
#define __nbits_aligned8(n) (((n) + 7) >> 3)
#endif
    
    typedef ptrdiff_t csint;
    
    typedef enum {
        ARMASSP_UNDEF_ORDER = 0,
        ARMASSP_COL_ORDER = 1,
        ARMASSP_ROW_ORDER = 2
    } armassp_order_t;

    typedef enum {
        ARMASSP_CSC = 0,
        ARMASSP_CSR = 1,
        ARMASSP_COO = 2
    } armassp_type_enum;
    
    enum armas_mmbits {
        ARMAS_MM_HEADER       = 0x1,
        ARMAS_MM_MATRIX       = 0x2,
        ARMAS_MM_REAL         = 0x4,
        ARMAS_MM_COMPLEX      = 0x8,
        ARMAS_MM_INTEGER      = 0x10,
        ARMAS_MM_ARRAY        = 0x20,
        ARMAS_MM_COORDINATE   = 0x40,
        ARMAS_MM_SYMMETRIC    = 0x80,
        ARMAS_MM_HERMITIAN    = 0x100,
        ARMAS_MM_GENERAL      = 0x200
    };
    
    typedef struct coo_elem {
        DTYPE val;
        int i;
        int j;
    } coo_elem_t;

    // TODO: change space allocation to request one big block; divide it 
    typedef struct armas_x_sparse_s {
        union {
            DTYPE *v;
            coo_elem_t *ep;
        } elems;
        int rows;               ///< Matrix row dimension
        int cols;               ///< Matrix columns
        int nnz;                ///< Number of non-zero elements
        int *ptr;               ///< Row/column pointers (size == rows+1 or cols+1) (null for COO)
        int *ix;                ///< Row/column indexes (size == nnz) (null for COO)
        int nptr;               ///< Number of elements in ptr array (== cols || rows)
        int size;               ///< Size of elements buffer
        armassp_type_enum kind;
        size_t __nbytes;
    } armas_x_sparse_t;

    typedef struct armas_x_spvec_s {
        DTYPE *elems;          ///< Vector elements
        int *ix;                ///< Indexes of non zero elements
        int nz;                 ///< Number of non-zero entries
    } armas_x_spvec_t;

    typedef struct armas_x_accum_s {
        DTYPE *elems;
        int *mark;
        int *queue;
        int nz;
        int tail;
    } armas_x_accum_t;

    __ARMAS_INLINE
    size_t armas_x_accum_bytes(int n) {
        return __align64(n*sizeof(DTYPE) + 2*n*sizeof(int));
    }

    __ARMAS_INLINE
    int armas_x_accum_dim(const armas_x_sparse_t *A) {
        return A->kind == ARMASSP_CSR ? A->cols : A->rows;
    }
        
    __ARMAS_INLINE
    size_t armas_x_accum_need(const armas_x_sparse_t *A) {
        return armas_x_accum_bytes(armas_x_accum_dim(A));
    }
    
    extern size_t armas_x_accum_make(armas_x_accum_t *acc, int n, void *ptr, size_t len);
    extern int armas_x_accum_allocate(armas_x_accum_t *acc, int n);
    extern void armas_x_accum_release(armas_x_accum_t *acc);
    extern void armas_x_accum_addpos(armas_x_accum_t *acc, int k, DTYPE v, int mark);
    extern void armas_x_accum_scatter(armas_x_accum_t *acc, const armas_x_spvec_t *x, DTYPE beta, int mark);
    extern void armas_x_accum_dot(armas_x_accum_t *acc, int k,
                                  const armas_x_spvec_t *x, const armas_x_spvec_t *y, int mark);
    extern void armas_x_accum_gather(armas_x_sparse_t *C, DTYPE alpha,
                                   armas_x_accum_t *acc, int ik, int maxnz);
    extern void armas_x_accum_clear(armas_x_accum_t *acc);



    typedef struct armassp_x_precond_s {
        armas_x_sparse_t *M;
        int flags;
        int (*precond)(armas_x_dense_t *z, const struct armassp_x_precond_s *M, const armas_x_dense_t *x, armas_conf_t *cf);
        int (*partial)(armas_x_dense_t *z, const struct armassp_x_precond_s *M, const armas_x_dense_t *x, int flags, armas_conf_t *cf);
    } armassp_x_precond_t;

    extern int armassp_x_init_iluz(armassp_x_precond_t *P, armas_x_sparse_t *A);
    extern int armassp_x_init_icholz(armassp_x_precond_t *P, armas_x_sparse_t *A, int flags);
    extern void armassp_x_precond_release(armassp_x_precond_t *P);
    
    // ------------------------------------------------------------------------------------
    // 
    __ARMAS_INLINE
    size_t armassp_x_bytes_needed(int rows, int cols, int nnz, armassp_type_enum kind)
    {
        if (kind == ARMASSP_COO)
            return nnz*sizeof(coo_elem_t);
        int n = kind == ARMASSP_CSR ? rows : cols;
        return __align64(nnz*sizeof(DTYPE) + sizeof(int)*(nnz + n+1));
    }

    __ARMAS_INLINE
    size_t armassp_x_bytes_for(const armas_x_sparse_t *A) 
    {
        return armassp_x_bytes_needed(A->rows, A->cols, A->nnz, A->kind);
    }

    __ARMAS_INLINE
    size_t armassp_x_nbytes(const armas_x_sparse_t *A)
    {
        return A->__nbytes;
    }

    // ------------------------------------------------------------------------------
    // marking the vertexes on index array


    // ------------------------------------------------------------------------------
    // sparse element access
    
    // \brief Get index pointer in graph.
    __ARMAS_INLINE
    int armassp_x_index(const armas_x_sparse_t *A, int j)
    {
        return A->ptr[j];
    }

#if 0    
    // \brief Get safely index position in graph. (never value < 0)
    __ARMAS_INLINE
    int armassp_x_index_safe(const armas_x_sparse_t *A, int j)
    {
        return sp_get_unmarked(A, j);
    }
#endif
    __ARMAS_INLINE
    int armassp_x_len(const armas_x_sparse_t *A, int j)
    {
        return A->ptr[j+1]-A->ptr[j];
    }
    // \brief Get index at position in graph.
    __ARMAS_INLINE
    int armassp_x_at(const armas_x_sparse_t *A, int i)
    {
        return A->ix[i];
    }

    // \brief Get indexes of non-zero values
    __ARMAS_INLINE
    int *armassp_x_iptr(const armas_x_sparse_t *A, int j)
    {
        return &A->ix[A->ptr[j]];
    }

    __ARMAS_INLINE
    int armassp_x_nz(const armas_x_sparse_t *A, int j, int ix)
    {
        for (int p = A->ptr[j]; A->ix[p] <= ix && p < A->ptr[j+1]; p++) {
            if (A->ix[p] == ix)
                return p;
        }
        return -1;
    }
    // \brief Get non-zero values
    __ARMAS_INLINE
    DTYPE *armassp_x_data(const armas_x_sparse_t *A, int j)
    {
        return &A->elems.v[armassp_x_index(A, j)];
    }

    __ARMAS_INLINE
    DTYPE armassp_x_value(const armas_x_sparse_t *A, int p)
    {
        return A->elems.v[p];
    }

    __ARMAS_INLINE
    int armassp_x_nvertex(const armas_x_sparse_t *A)
    {
        return A->nptr;
    }

    __ARMAS_INLINE
    int armassp_x_size(const armas_x_sparse_t *A)
    {
        return A ? A->rows * A->cols : 0;
    }

    __ARMAS_INLINE
    armas_x_spvec_t *armassp_x_vector(armas_x_spvec_t *x, const armas_x_sparse_t *A, int k)
    {
        x->elems = armassp_x_data(A, k);
        x->ix = armassp_x_iptr(A, k);
        x->nz = armassp_x_len(A, k);
        return x;
    }

    __ARMAS_INLINE
    armas_x_sparse_t *armassp_x_clear(armas_x_sparse_t *A) {
        A->ptr = A->ix = (int *)0;
        A->rows = A->cols = A->nnz = A->size = A->nptr = 0;
        A->elems.v = (DTYPE *)0;
        A->__nbytes = 0;
        return A;
    }

    __ARMAS_INLINE
    void armassp_x_release(armas_x_sparse_t *A) {
        if (A->elems.v)
            free(A->elems.v);
        armassp_x_clear(A);
    }
    
    __ARMAS_INLINE
    void armassp_x_free(armas_x_sparse_t *A)
    {
        if (A) {
            armassp_x_release(A);
            free(A);
        }
    }

    
    extern int armassp_x_make(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum storage, void *data, size_t dlen);
    extern armas_x_sparse_t *armassp_x_init(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum kind);
    extern armas_x_sparse_t *armassp_x_new(int rows, int cols, int nnz, armassp_type_enum kind);
    extern armas_x_sparse_t *armassp_x_spa_alloc(armas_x_sparse_t *A, int nnz);
    extern int armassp_x_append(armas_x_sparse_t *A, int row, int col, DTYPE val);
    extern int armassp_x_resize(armas_x_sparse_t *A, int newsize);
    extern int armassp_x_sort_to(armas_x_sparse_t *A, armassp_order_t order);
    extern int armassp_x_sort(armas_x_sparse_t *A);

    extern armas_x_sparse_t *armassp_x_convert(const armas_x_sparse_t *B, armassp_type_enum target);
    extern armas_x_sparse_t *armassp_x_convert_to(armas_x_sparse_t *A, const armas_x_sparse_t *B, armassp_type_enum target);
    extern armas_x_sparse_t *armassp_x_transpose(const armas_x_sparse_t *B);
    extern armas_x_sparse_t *armassp_x_transpose_to(armas_x_sparse_t *A, const armas_x_sparse_t *B);
    extern armas_x_sparse_t *armassp_x_copy_to(armas_x_sparse_t *A, const armas_x_sparse_t *B);
    extern armas_x_sparse_t *armassp_x_mkcopy(const armas_x_sparse_t *B);

    extern armas_x_sparse_t *armassp_x_mmload(int *typecode, FILE *f);
    extern int armassp_x_mmdump(FILE *f, const armas_x_sparse_t *A, int flags);
    extern void armassp_x_pprintf(FILE *f, const armas_x_sparse_t *A);
    extern void armassp_x_iprintf(FILE *f, const armas_x_sparse_t *A);
    extern int armassp_x_todense(armas_x_dense_t *A, const armas_x_sparse_t *B, armas_conf_t *cf);

    extern int armassp_x_mvmult_trm(armas_x_dense_t *x, DTYPE alpha,
                                    const armas_x_sparse_t *A, int flags, armas_conf_t *cf);
    extern int armassp_x_mvsolve_trm(armas_x_dense_t *x, DTYPE alpha,
                                     const armas_x_sparse_t *A, int flags, armas_conf_t *cf);
    extern int armassp_x_mvmult(DTYPE beta, armas_x_dense_t *y, DTYPE alpha,
                                const armas_x_sparse_t *A, const armas_x_dense_t *x,
                                int flags, armas_conf_t *cf);
    extern int armassp_x_mvmult_sym(DTYPE beta, armas_x_dense_t *y, DTYPE alpha,
                                    const armas_x_sparse_t *A, const armas_x_dense_t *x,
                                    int flags, armas_conf_t *cf);
    
    extern int armassp_x_cgrad(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                               int flags, armas_conf_t *cf);
    extern int armassp_x_cgrad_w(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                                 int flags, armas_wbuf_t *W, armas_conf_t *cf);
    extern int armassp_x_pcgrad_w(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                                  armassp_x_precond_t *P, int flags, armas_wbuf_t *W, armas_conf_t *cf);
    extern int armassp_x_pcgrad(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                                armassp_x_precond_t *P, int flags, armas_conf_t *cf);

    extern int armassp_x_cgnr(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                              armas_conf_t *cf);
    extern int armassp_x_cgnr_w(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                                armas_wbuf_t *W, armas_conf_t *cf);
    extern int armassp_x_cgne(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                              armas_conf_t *cf);
    extern int armassp_x_cgne_w(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                                armas_wbuf_t *W, armas_conf_t *cf);
    extern int armassp_x_gmres(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                               armas_conf_t *cf);
    extern int armassp_x_gmres_w(armas_x_dense_t *x,  const armas_x_sparse_t *A,
                                 const armas_x_dense_t *b, armas_wbuf_t *W, armas_conf_t *cf);
    extern int armassp_x_pgmres(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
                                const armassp_x_precond_t *M, armas_conf_t *cf);
    extern int armassp_x_pgmres_w(armas_x_dense_t *x,  const armas_x_sparse_t *A,
                                  const armas_x_dense_t *b, const armassp_x_precond_t *M,
                                  armas_wbuf_t *W, armas_conf_t *cf);

    extern int armassp_x_addto_w(armas_x_sparse_t *C, DTYPE alpha, const armas_x_sparse_t *A,
                                 DTYPE beta, const armas_x_sparse_t *B, int bits, armas_wbuf_t *work,
                                 armas_conf_t *cf);
    extern armas_x_sparse_t *armassp_x_add(DTYPE alpha, const armas_x_sparse_t *A,
                                           DTYPE beta, const armas_x_sparse_t *B, int bits, 
                                           armas_conf_t *cf);

    extern int armassp_x_multto_w(armas_x_sparse_t *C,
                                  DTYPE alpha, const armas_x_sparse_t *A, const armas_x_sparse_t *B,
                                  int bits, armas_wbuf_t *work, armas_conf_t *cf);
    extern armas_x_sparse_t *armassp_x_mult(DTYPE alpha, const armas_x_sparse_t *A,
                                            const armas_x_sparse_t *B, int bits, armas_conf_t *cf);
    

    extern int armassp_x_init_icholz(armassp_x_precond_t *P, armas_x_sparse_t *A, int flags);
    extern int armassp_x_icholz(armas_x_sparse_t *A, int flags);
    extern int armassp_x_iluz(armas_x_sparse_t *L);

    extern int armassp_x_hasdiag(const armas_x_sparse_t *A, int diag);
    
#ifdef __cplusplus
}
#endif
  
#endif

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
  
