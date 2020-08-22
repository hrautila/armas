
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_MATCPY_H
#define ARMAS_MATCPY_H 1


#include <string.h>

static inline
void copy_plain_mcpy1(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int j;
    for (j = 0; j < nC; j ++) {
        memcpy(&d[(j+0)*ldD], &s[(j+0)*ldS], nR*sizeof(DTYPE));
    }
}

static inline
void copy_plain_abs(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC-3; j += 4) {
        for (i = 0; i < nR; i++) {
            d[(i+0)+(j+0)*ldD] = ABS(s[(i+0)+(j+0)*ldS]);
            d[(i+0)+(j+1)*ldD] = ABS(s[(i+0)+(j+1)*ldS]);
            d[(i+0)+(j+2)*ldD] = ABS(s[(i+0)+(j+2)*ldS]);
            d[(i+0)+(j+3)*ldD] = ABS(s[(i+0)+(j+3)*ldS]);
        }
    }
    for (; j < nC; j++) {
        for (i = 0; i < nR; i++) {
            d[(i+0)+(j+0)*ldD] = ABS(s[(i+0)+(j+0)*ldS]);
        }
    }
}

static inline
void copy_trans1x4(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC; j ++) {
        for (i = 0; i < nR-3; i += 4) {
            d[(j+0)+(i+0)*ldD] = s[(i+0)+(j+0)*ldS];
            d[(j+0)+(i+1)*ldD] = s[(i+1)+(j+0)*ldS];
            d[(j+0)+(i+2)*ldD] = s[(i+2)+(j+0)*ldS];
            d[(j+0)+(i+3)*ldD] = s[(i+3)+(j+0)*ldS];
        }
        if (i == nR)
            continue;
        switch (nR-i) {
        case 3:
            d[j+i*ldD] = s[i+j*ldS];
            i++;
        case 2:
            d[j+i*ldD] = s[i+j*ldS];
            i++;
        case 1:
            d[j+i*ldD] = s[i+j*ldS];
        }
    }
}

static inline
void copy_trans4x1(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC-3; j += 4) {
        for (i = 0; i < nR; i ++) {
            d[(j+0)+(i+0)*ldD] = s[i+(j+0)*ldS];
            d[(j+1)+(i+0)*ldD] = s[i+(j+1)*ldS];
            d[(j+2)+(i+0)*ldD] = s[i+(j+2)*ldS];
            d[(j+3)+(i+0)*ldD] = s[i+(j+3)*ldS];
        }
    }
    if (j == nC)
        return;
    copy_trans1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

static inline
void copy_trans1x4_abs(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC; j ++) {
        for (i = 0; i < nR-3; i += 4) {
            d[(j+0)+(i+0)*ldD] = ABS(s[(i+0)+(j+0)*ldS]);
            d[(j+0)+(i+1)*ldD] = ABS(s[(i+1)+(j+0)*ldS]);
            d[(j+0)+(i+2)*ldD] = ABS(s[(i+2)+(j+0)*ldS]);
            d[(j+0)+(i+3)*ldD] = ABS(s[(i+3)+(j+0)*ldS]);
        }
        if (i == nR)
            continue;
        switch (nR-i) {
        case 3:
            d[j+i*ldD] = ABS(s[i+j*ldS]);
            i++;
        case 2:
            d[j+i*ldD] = ABS(s[i+j*ldS]);
            i++;
        case 1:
            d[j+i*ldD] = ABS(s[i+j*ldS]);
        }
    }
}

static inline
void copy_trans4x1_abs(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC-3; j += 4) {
        for (i = 0; i < nR; i ++) {
            d[(j+0)+(i+0)*ldD] = ABS(s[i+(j+0)*ldS]);
            d[(j+1)+(i+0)*ldD] = ABS(s[i+(j+1)*ldS]);
            d[(j+2)+(i+0)*ldD] = ABS(s[i+(j+2)*ldS]);
            d[(j+3)+(i+0)*ldD] = ABS(s[i+(j+3)*ldS]);
        }
    }
    if (j == nC)
        return;
    copy_trans1x4_abs(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

// Copy upper tridiagonal and fill lower part to form full symmetric matrix
// result is symmetric matrix A and A = A.T
static inline
void colcpy_fill_low(DTYPE *dst, int ldD, const DTYPE *src, int ldS, int nR, int nC, int unit)
{
    //assert(nR == nC);
    register int j, i;

    // fill dst row and column at the same time, following src columns
    for (j = 0; j < nC; j++) {
        for (i = 0; i < j; i++) {
            dst[i + j*ldD] = src[i + j*ldS];
            dst[j + i*ldD] = src[i + j*ldS];
        }
        // copy the diagonal entry
        dst[j + j*ldD] = unit ? 1.0 : src[j+j*ldS];
    }
}

// Copy lower tridiagonal and fill upper part to form full symmetric matrix;
// result is symmetric matrix A and A = A.T
static inline
void colcpy_fill_up(DTYPE *dst, int ldD, const DTYPE *src, int ldS, int nR, int nC, int unit)
{
    //assert(nR == nC);
    register int j, i;

    // fill dst row and column at the same time, following src columns
    for (j = 0; j < nC; j++) {
        dst[j + j*ldD] = unit ? 1.0 : src[j + j*ldS];

        // off diagonal entries
        for (i = j+1; i < nC; i++) {
            dst[i + j*ldD] = src[i + j*ldS];
            dst[j + i*ldD] = src[i + j*ldS];
        }
    }
}

// Copy upper tridiagonal and fill lower part to form full symmetric matrix
// result is symmetric matrix A and A = A.T
static inline
void colcpy_fill_low_abs(DTYPE *dst, int ldD, const DTYPE *src, int ldS, int nR, int nC, int unit)
{
    //assert(nR == nC);
    register int j, i;

    // fill dst row and column at the same time, following src columns
    for (j = 0; j < nC; j++) {
        for (i = 0; i < j; i++) {
            dst[i + j*ldD] = ABS(src[i + j*ldS]);
            dst[j + i*ldD] = ABS(src[i + j*ldS]);
        }
        // copy the diagonal entry
        dst[j + j*ldD] = unit ? 1.0 : ABS(src[j+j*ldS]);
    }
}

// Copy lower tridiagonal and fill upper part to form full symmetric matrix;
// result is symmetric matrix A and A = A.T
static inline
void colcpy_fill_up_abs(DTYPE *dst, int ldD, const DTYPE *src, int ldS, int nR, int nC, int unit)
{
    //assert(nR == nC);
    register int j, i;

    // fill dst row and column at the same time, following src columns
    for (j = 0; j < nC; j++) {
        dst[j + j*ldD] = unit ? 1.0 : ABS(src[j + j*ldS]);

        // off diagonal entries
        for (i = j+1; i < nC; i++) {
            dst[i + j*ldD] = ABS(src[i + j*ldS]);
            dst[j + i*ldD] = ABS(src[i + j*ldS]);
        }
    }
}




#if defined(COMPLEX64) || defined(COMPLEX128)
#include <complex.h>
#define CONJ(a) conj(a)

static inline
void copy_trans_conj1x4(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC; j ++) {
        for (i = 0; i < nR-3; i += 4) {
            d[(j+0)+(i+0)*ldD] = CONJ(s[(i+0)+(j+0)*ldS]);
            d[(j+0)+(i+1)*ldD] = CONJ(s[(i+1)+(j+0)*ldS]);
            d[(j+0)+(i+2)*ldD] = CONJ(s[(i+2)+(j+0)*ldS]);
            d[(j+0)+(i+3)*ldD] = CONJ(s[(i+3)+(j+0)*ldS]);
        }
        if (i == nR)
            continue;
        switch (nR-i) {
        case 3:
            d[j+i*ldD] = CONJ(s[i+j*ldS]);
            i++;
        case 2:
            d[j+i*ldD] = CONJ(s[i+j*ldS]);
            i++;
        case 1:
            d[j+i*ldD] = CONJ(s[i+j*ldS]);
        }
    }
}

static inline
void copy_trans_conj4x1(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC-3; j += 4) {
        for (i = 0; i < nR; i ++) {
            d[(j+0)+(i+0)*ldD] = CONJ(s[i+(j+0)*ldS]);
            d[(j+1)+(i+0)*ldD] = CONJ(s[i+(j+1)*ldS]);
            d[(j+2)+(i+0)*ldD] = CONJ(s[i+(j+2)*ldS]);
            d[(j+3)+(i+0)*ldD] = CONJ(s[i+(j+3)*ldS]);
        }
    }
    if (j == nC)
        return;
    copy_trans_conj1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

static inline
void copy_conj1x4(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC; j ++) {
        for (i = 0; i < nR-3; i += 4) {
            d[(i+0)+(j+0)*ldD] = CONJ(s[(i+0)+(j+0)*ldS]);
            d[(i+1)+(j+0)*ldD] = CONJ(s[(i+1)+(j+0)*ldS]);
            d[(i+2)+(j+0)*ldD] = CONJ(s[(i+2)+(j+0)*ldS]);
            d[(i+3)+(j+0)*ldD] = CONJ(s[(i+3)+(j+0)*ldS]);
        }
        if (i == nR)
            continue;
        switch (nR-i) {
        case 3:
            d[i+j*ldD] = CONJ(s[i+j*ldS]);
            i++;
        case 2:
            d[i+j*ldD] = CONJ(s[i+j*ldS]);
            i++;
        case 1:
            d[i+j*ldD] = CONJ(s[i+j*ldS]);
        }
    }
}

static inline
void copy_conj4x1(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    register int i, j;
    for (j = 0; j < nC-3; j += 4) {
        for (i = 0; i < nR; i ++) {
            d[i+(j+0)*ldD] = CONJ(s[i+(j+0)*ldS]);
            d[i+(j+1)*ldD] = CONJ(s[i+(j+1)*ldS]);
            d[i+(j+2)*ldD] = CONJ(s[i+(j+2)*ldS]);
            d[i+(j+3)*ldD] = CONJ(s[i+(j+3)*ldS]);
        }
    }
    if (j == nC)
        return;
    copy_conj1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

static inline
void CPBLK_TRANS(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_CTRANSA|ARMAS_CTRANSB|ARMAS_CONJA|ARMAS_CONJB)) {
        copy_trans_conj4x1(d->elems, d->step, s->elems, s->step, nR, nC);
    } else {
        copy_trans4x1(d->elems, d->step, s->elems, s->step, nR, nC);
    }
}

static inline
void CPBLK(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_CTRANSA|ARMAS_CTRANSB|ARMAS_CONJA|ARMAS_CONJB)) {
        copy_conj4x1(d->elems, d->step, s->elems, s->step, nR, nC);
    } else {
        copy_plain_mcpy1(d->elems, d->step, s->elems, s->step, nR, nC);
    }
}

static inline
void CPBLK_TRIL_UFILL(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_CTRANSA|ARMAS_CTRANSB|ARMAS_CONJA|ARMAS_CONJB)) {
    } else {
        colcpy_fill_up(d->elems, d->step, s->elems, s->step, nR, nC, (flags&ARMAS_UNIT));
    }
}

static inline
void CPBLK_TRIU_LFILL(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_CTRANSA|ARMAS_CTRANSB|ARMAS_CONJA|ARMAS_CONJB)) {
    } else {
        colcpy_fill_low(d->elems, d->step, s->elems, s->step, nR, nC, (flags&ARMAS_UNIT));
    }
}
#else  //defined(FLOAT32) || defined(FLOAT64)

static inline
void CPTRANS(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    copy_trans4x1(d, ldD, s, ldS, nR, nC);
}

static inline
void CP(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC)
{
    copy_plain_mcpy1(d, ldD, s, ldS, nR, nC);
}

static inline
void CPTRIL_UFILL(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int unit)
{
    colcpy_fill_up(d->elems, d->step, s->elems, s->step, nR, nC, unit);
}

static inline
void CPTRIU_LFILL(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int unit)
{
    colcpy_fill_low(d->elems, d->step, s->elems, s->step, nR, nC, unit);
}

static inline
void CPBLK_TRANS(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_ABSA|ARMAS_ABSB)) {
        copy_trans4x1_abs(d->elems, d->step, s->elems, s->step, nR, nC);
    } else {
        copy_trans4x1(d->elems, d->step, s->elems, s->step, nR, nC);
    }
}

static inline
void CPBLK(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if ((flags & (ARMAS_ABSA|ARMAS_ABSB)) != 0) {
        copy_plain_abs(d->elems, d->step, s->elems, s->step, nR, nC);
    } else {
        copy_plain_mcpy1(d->elems, d->step, s->elems, s->step, nR, nC);
    }
}

static inline
void CPBLK_TRIL_UFILL(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_ABSA|ARMAS_ABSB)) {
        colcpy_fill_up_abs(d->elems, d->step, s->elems, s->step, nR, nC, (flags&ARMAS_UNIT));
    } else {
        colcpy_fill_up(d->elems, d->step, s->elems, s->step, nR, nC, (flags&ARMAS_UNIT));
    }
}

static inline
void CPBLK_TRIU_LFILL(armas_x_dense_t *d, const armas_x_dense_t *s, int nR, int nC, int flags)
{
    if (flags & (ARMAS_ABSA|ARMAS_ABSB)) {
        colcpy_fill_low_abs(d->elems, d->step, s->elems, s->step, nR, nC, (flags&ARMAS_UNIT));
    } else {
        colcpy_fill_low(d->elems, d->step, s->elems, s->step, nR, nC, (flags&ARMAS_UNIT));
    }
}

#endif  /* FLOAT32 && FLOAT64*/

static inline
void blk_scale(armas_x_dense_t *x, const DTYPE beta, int M, int N)
{
    register DTYPE *X = x->elems;
    register int ldX = x->step;
    register int i, j;

    if (beta == 1.0) {
        return;
    }

    // set to zero
    if (beta == 0.0) {
        for (j = 0; j < N-3; j += 4) {
            for (i = 0; i < M; i++) {
                X[i+(j+0)*ldX] = 0.0;
                X[i+(j+1)*ldX] = 0.0;
                X[i+(j+2)*ldX] = 0.0;
                X[i+(j+3)*ldX] = 0.0;
            }
        }
        if (j == N)
            return;
        for (; j < N; j++) {
            for (i = 0; i < M; i++) {
                X[i+(j+0)*ldX] = 0.0;
            }
        }
        return;
    }
    // scale here
    for (j = 0; j < N-3; j += 4) {
        for (i = 0; i < M; i++) {
            X[i+(j+0)*ldX] *= beta;
            X[i+(j+1)*ldX] *= beta;
            X[i+(j+2)*ldX] *= beta;
            X[i+(j+3)*ldX] *= beta;
        }
    }
    if (j == N)
        return;
    for (; j < N; j++) {
        for (i = 0; i < M; i++) {
            X[i+(j+0)*ldX] *= beta;
        }
    }
}

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
