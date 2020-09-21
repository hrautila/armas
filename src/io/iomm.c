// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "dtype.h"

// ----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mmload) && defined(armas_x_mmdump)
#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
//! \endcond


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

#if COMPLEX32 || COMPLEX64
#define MM_DTYPE ARMAS_MM_COMPLEX
#define TYPE_STRING "complex"

static inline
DTYPE read_value(char *s)
{
    char *endp;
    ABSTYPE a, b;
    a = strtod(s, &endp);
    s = *endp;
    b = strtod(s, &endp);
    return (DTYPE)(a + b*I);
}


#else
#define MM_DTYPE ARMAS_MM_REAL
#define TYPE_STRING "real"

#if FLOAT32
#define FMT1 "%f"
#define FMT3 "%d %d %f"
#else
#define FMT1 "%lf"
#define FMT3 "%d %d %lf"
#endif

static inline
int read_value(const char *s, DTYPE *val)
{
    return sscanf(s, FMT1, val);
}

static inline
int read_line(const char *s, int *row, int *col, DTYPE *val)
{
    return sscanf(s, FMT3, row, col, val);
}

#endif

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

/*
 * Matrix market Array Format for dense matrices
 * 
 * 1.  | %%matrixMarket matrix array real general
 * 2.  | % comment
 * 3.  | 2 3        ! size [rows, cols]
 * 4.  | 1.0        ! first entry
 * 5.  | 2.0
 * 6.  | 3.0        !
 *       ...
 * 9.  | 6.0        ! last entry
 *

 * Matrix market Coordinate Format for dense matrices
 * 
 * 1.  | %%matrixMarket matrix coordinate real general
 * 2.  | % comment
 * 3.  | 2 3 6      ! size [rows, cols, nnz]
 * 4.  | 0 0 1.0    ! first entry
 * 5.  | 1 0 2.0
 * 6.  | 0 1 3.0    !
 *       ...
 * 9.  | 1 2 6.0    ! last entry

 */
static inline
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

static
int __mmread_banner(FILE *f, int *typecode)
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
            if (nc != ARMAS_MM_REAL && nc != ARMAS_MM_COMPLEX)
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
static
int __mmread_size(FILE *f, int typecode, int *m, int *n, int *nnz)
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

/**
 * @brief Load matrix from a matrix market format file
 *
 * @param[out] A
 *    On entry uninitialized matrix. On exit matrix loaded from file.
 * @param[out] flags
 *    On exit file storage layout indicators, zero for general matrix, ARMAS_SYMM
 *    for symmetric matrix and only lower triangular part is stored.
 * @param[in] f
 *    Pointer to opened file. Reading starts at current file location.
 *
 * @returns
 *    0 on success
 *    <0 on error
 * 
 * @ingroup matrix
 */
int armas_x_mmload(armas_x_dense_t *A, int *flags,  FILE *f)
{
    int typecode;
    int m, n, nc, nnz, r, c, nelem, nread, k;
    char *iobuf;
    size_t ioblen;
    double v;

    if (__mmread_banner(f, &typecode) < 0) {
        return -1;
    }
    // MM_DTYPE is element type 
    if ((typecode & ARMAS_MM_MATRIX) == 0 ||
        (typecode & MM_DTYPE) == 0) {
        // must be matrix in array format with matching element type
        return -1;
    }
    if (__mmread_size(f, typecode, &m, &n, &nnz) < 0) {
        return -1;
    }

    if ((typecode & ARMAS_MM_SYMMETRIC) != 0 && m != n) {
        return -1;
    }

    armas_x_init(A, m, n);

    // number of elements to expect
    nelem = nnz;
    if ((typecode & ARMAS_MM_ARRAY) != 0) 
        nelem = (typecode & ARMAS_MM_SYMMETRIC) != 0 ? n*(n + 1)/2 : m*n;

    ioblen = 0;
    iobuf = (char *)0;
    c =  k = 0;
    r = -1;
    nc = m;  // for lower triangular storage index for next column
    while ((nread = getline(&iobuf, &ioblen, f)) != -1 && k < nelem) {
        if (*iobuf == '%')
            continue;
        if ((typecode & ARMAS_MM_ARRAY) != 0) {
            //v = strtod(iobuf, &endptr);
            if (read_value(iobuf, &v) < 0)
                goto endloop;

            if ((typecode & ARMAS_MM_SYMMETRIC) != 0) {
                // only lower triangular elements provided (n*(n+1)/2 elements)
                // in packed format
                if (k == nc) {
                    // next column, update start index for following column
                    c++;
                    nc += m - c;
                    r = c - 1;
                }
                r++;
            }
            else {
                r = k%m;
                c = k/m;
            }
        }
        else {
            if (read_line(iobuf, &r, &c, &v) < 0)
                goto endloop;
        }
        armas_x_set_unsafe(A, r, c, v);
        k++;
    }
 endloop:
    // getline allocates; we free it
    if (iobuf)
        free(iobuf);

    // assert k == m*n; for general matrix; k == n*(n+1)/2 for lower triangular
    if (k != nelem)
        return -1;

    return typecode;
}


int armas_x_mmdump(FILE *f, const armas_x_dense_t *A, int flags)
{
    char *s = "general";
    char *t = TYPE_STRING;
    int n, j, k;

    if  ((flags & ARMAS_SYMM) != 0)
        s = "symmetric";

    n  = fprintf(f, "%%%%Matrixmarket matrix array %s %s\n", t, s);
    n += fprintf(f, "%d %d\n", A->rows, A->cols);

    for (j = 0; j < A->cols; j++) {
        k = (flags & ARMAS_SYMM) != 0 ? j : 0;
        for ( ; k < A->rows; k++) {
            n += fprintf(f, "%.13e\n", armas_x_get_unsafe(A, k, j));
        }
    }
    return n;
}
#else
#warning "Missing defines! No code."
#endif // defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
