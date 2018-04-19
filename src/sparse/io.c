
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_mmload) && defined(armassp_x_mmdump)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_new)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "matrix.h"
#include "sparse.h"

#define MM_HEADER       0x1
#define MM_MATRIX       0x2
#define MM_REAL         0x4
#define MM_COMPLEX      0x8
#define MM_INTEGER      0x10
#define MM_ARRAY        0x20
#define MM_COORDINATE   0x40
#define MM_SYMMETRIC    0x80
#define MM_HERMITIAN    0x100
#define MM_GENERAL      0x200

struct mm_keys {
    char *str;
    int slen;
    int code;
};
static struct mm_keys __mmkeys[] = (struct mm_keys []) {
    { "matrix",           7, ARMAS_MM_MATRIX },
    { "real",             5, ARMAS_MM_REAL },
    { "complex",          8, ARMAS_MM_COMPLEX },
    { "integer",          8, ARMAS_MM_INTEGER },
    { "array",            6, ARMAS_MM_ARRAY },
    { "coordinate",      11, ARMAS_MM_COORDINATE },
    { "symmetric",       10, ARMAS_MM_SYMMETRIC },
    { "hermitian",       10, ARMAS_MM_HERMITIAN },
    { "general",          7, ARMAS_MM_GENERAL },
    { "%%matrixmarket",  15, ARMAS_MM_HEADER },
    { 0, 0, 0}
};

static
int find_key(const char *key)
{
    for (int k = 0; __mmkeys[k].slen != 0; k++) {
        if (strncmp(key, __mmkeys[k].str, __mmkeys[k].slen) == 0)
            return __mmkeys[k].code;
    }
    return 0;
}

static inline
char *strlower(char *s)
{
    for (char *cp = s; *cp; cp++)
        *cp = tolower(*cp);
    return s;
}

/*
 *  banner row:
 *  %%Matrixmarket matrix|image array|coordinate real|complex|integer  <empty>|symmetric|hermitian
 */

int armasio_mmread_banner(FILE *f, int *typecode)
{
    char iob[512];
    char *s, *tok;
    int code, n, nc;

    if (!fgets(iob, sizeof(iob), f))
        return -1;
    if (iob[0] != '%' && iob[1] != '%')
        return -1;

    if ((s = strchr(iob, '\n'))) {
        *s = '\0';
    }

    int ready = 0;
    code = 0;
    s = iob;
    for (n = 0, tok = strsep(&s, " "); tok && ready == 0; tok = strsep(&s, " "), n++) {
        nc = find_key(strlower(tok));
        switch (n) {
        case 0:
            // %%MatrixMarket
            if (nc != ARMAS_MM_HEADER)
                return -1;
            break;
        case 1:
            // object
            if (nc != ARMAS_MM_MATRIX)
                return -1;
            code |= nc;
            break;
        case 2:
            // storage format (coordinate, array)
            if (nc != ARMAS_MM_ARRAY && nc != ARMAS_MM_COORDINATE)
                return -1;
            code |= nc;
            break;
        case 3:
            // element type (real, complex, pattern, integer)
            if (nc != ARMAS_MM_REAL)
                return -1;
            code |= nc;
            break;
        case 4:
            // symmetries (symmetric, skew-symmetric, hermitian)
            if (nc != ARMAS_MM_SYMMETRIC && nc != ARMAS_MM_HERMITIAN && nc != ARMAS_MM_GENERAL)
                return -1;
            code |= nc;
        default:
            ready = 1;
            break;
        }
    }
    *typecode = code;
    return 0;
}

/*
 * size row:
 * nrows ncols number-of-non-zeros|<empty>
 */
int armasio_mmread_size(FILE *f, int typecode, int *m, int *n, int *nnz)
{
    char *iob = (char *)0;
    char *cp, *endptr;
    int nread;
    size_t ioblen = 0;

    while ((nread = getline(&iob, &ioblen, f)) != -1) {
        if (iob[0] != '%')
            break;
    }
    if (isspace(iob[0]) || isdigit(iob[0])) {
        cp = iob;
        *m = strtol(cp, &endptr, 10);
        cp = endptr + 1;
        *n = strtol(cp, &endptr, 10);
        if ((typecode & ARMAS_MM_COORDINATE) != 0) {
            cp = endptr + 1;
            *nnz =strtol(cp, &endptr, 10);
        } else {
            *nnz = 0;
        }
    } else {
        *m = *n = *nnz = 0;
    }
    free(iob);
    return 0;
}

armas_x_sparse_t *armassp_x_mmload(int *typecode, FILE *f)
{
    int tc;
    int m, n, nnz;
    char *iobuf, *bp, *endptr;
    size_t ioblen;
    armas_x_sparse_t *A;
    
    if (armasio_mmread_banner(f, &tc) < 0) {
        return (armas_x_sparse_t *)0;
    }
    *typecode = tc;
    if ((tc & ARMAS_MM_MATRIX) == 0 ||
        (tc & ARMAS_MM_REAL) == 0 ||
        (tc & ARMAS_MM_COORDINATE) == 0) {
        // must be real matrix in coordinate format
        return (armas_x_sparse_t *)0;
    }
    if (armasio_mmread_size(f, tc, &m, &n, &nnz) < 0) {
        return (armas_x_sparse_t *)0;
    }

    A = armassp_x_new(m, n, nnz, ARMASSP_COO);
    if (!A)
        return A;
    
    ioblen = 0; iobuf = (char *)0;
    int nread, k = 0;
    double v;
    while ((nread = getline(&iobuf, &ioblen, f)) != -1 && k < nnz) {
        bp = iobuf;
        m = strtol(bp, &endptr, 10);
        bp = endptr + 1;
        n = strtol(bp, &endptr, 10);
        bp = endptr + 1;
        v = strtod(bp, &endptr);
        armassp_x_append(A, m-1, n-1, v);
#if 0
        if ((tc & ARMAS_MM_SYMMETRIC) != 0 && m != n) {
            armassp_x_append(A, n-1, m-1, v);
        }
#endif
        k++;
    }
    // getline allocates; we free it
    if (iobuf)
        free(iobuf);

    return A;
}

int armassp_x_mmdump(FILE *f, const armas_x_sparse_t *A, int flags)
{
    int n;
    char *s = "general";

    if ( (flags&MM_SYMMETRIC) != 0)
        s = "symmetric";

    switch (A->kind) {
    case ARMASSP_COO:
        n  = fprintf(f, "%%%%Matrixmarket matrix coordinate real %s\n", s);
        n += fprintf(f, "%d %d %d\n", A->rows, A->cols, A->nnz);
        for (int k = 0; k < A->nnz; k++) {
            n += fprintf(f, "%d %d %.13e\n", A->elems.ep[k].i, A->elems.ep[k].j, A->elems.ep[k].val);
        }
        break;
    case ARMASSP_CSR:
        n  = fprintf(f, "%%%%Matrixmarket matrix coordinate real %s\n", s);
        n += fprintf(f, "%d %d %d\n", A->rows, A->cols, A->nnz);
        for (int i = 0; i < A->rows; i++) {
            for (int k = A->ptr[i]; k < A->ptr[i+1]; k++) {
                n += fprintf(f, "%d %d %.13e\n", i, A->ix[k], A->elems.v[k]);
            }
        }
        break;
    case ARMASSP_CSC:
        n  = fprintf(f, "%%%%Matrixmarket matrix coordinate real %s\n", s);
        n += fprintf(f, "%d %d %d\n", A->rows, A->cols, A->nnz);
        for (int j = 0; j < A->cols; j++) {
            for (int k = A->ptr[j]; k < A->ptr[j+1]; k++) {
                n += fprintf(f, "%d %d %.13e\n", A->ix[k], j, A->elems.v[k]);
            }
        }
        break;
    default:
        n = 0;
        break;
    }
    return n;
}

#if defined(armassp_x_pprintf)
void armassp_x_pprintf(FILE *f, const armas_x_sparse_t *A)
{
    coo_elem_t *Ae;
    char *s = malloc(A->rows*A->cols);
    if (!s)
        return;
    memset(s, '.', A->rows*A->cols);
    switch (A->kind) {
    case ARMASSP_COO:
        Ae = A->elems.ep;
        for (int k = 0; k < A->nnz; k++) {
            s[Ae[k].i + A->rows*Ae[k].j] = 'x';
        }
        break;
    case ARMASSP_CSR:
        for (int j = 0; j < A->rows; j++) {
            for (int i = A->ptr[j]; i < A->ptr[j+1]; i++) {
                s[j + A->rows*A->ix[i]] = 'x';
            }       
        }
        break;
    case ARMASSP_CSC:
        for (int j = 0; j < A->cols; j++) {
            for (int i = A->ptr[j]; i < A->ptr[j+1]; i++) {
                s[A->ix[i] + A->rows*j] = 'x';
            }       
        }
        break;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            fputc(s[i + A->rows*j], f);
        }
        fputc('\n', f);
    }
    free(s);
}
#endif

#if defined(armassp_x_iprintf)
void armassp_x_iprintf(FILE *f, const armas_x_sparse_t *A)
{
    if (A->kind == ARMASSP_COO)
        return;
    
    char *t = A->kind == ARMASSP_CSC ? "c" : "r";
    for (int k = 0; k < A->nptr; k++) {
        fprintf(f, "%c%02d: [", *t, k);
        for (int i = A->ptr[k]; i < A->ptr[k+1]; i++) {
            if (i != A->ptr[k])
                fprintf(f, ",");
            fprintf(f, "%3d", A->ix[i]);
                    
        }
        fprintf(f, "]\n");
    }
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
