

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>

// Generate ill-conditioned vectors for dot product with targeted condition number.
// 
// For details see algorigh 6.1 in 
//     T. Ogita, S. Rump, S. Oishi
//     Accurate Sum and Dot Product,
//     2005, SIAM Journal of Scientific Computing
//  
typedef double DTYPE;
#include "eft.h"
#include "testing.h"

#include <gmp.h>
#include <mpfr.h>

#if 0
extern double prec_dot(double *, double *, double, int, int);
extern void prec_axpy(double *, double *, double, double, int, int);
extern double prec_nrm2(double *, double, int, int);
extern double prec_sum(double *, double, int, int);
extern double prec_asum(double *, double, int, int);
extern void prec_gemv(double *y, double *A, double *x, double alpha, double beta, int N, int prec, int trans);
extern void prec_gemm(double *y, double *A, double *x, double alpha, double beta, int N, int prec, int trans);

extern void prec_gendot(double *dot, double *tcond, double *x, double *y, int N, double cond);
extern void prec_gensum(double *dot, double *tcond, double *x, int N, double cond);

// extended precision helpers functions; require mpfr-library.
extern double ep_dot_n(armas_d_dense_t *X, armas_d_dense_t *Y, double start, int N, int prec);
extern double ep_dot(armas_d_dense_t *X, armas_d_dense_t *Y, double start, int prec);
extern double ep_nrm2(armas_d_dense_t *X, double start, int prec);
extern double ep_sum(armas_d_dense_t *X, double start, int prec);
extern double ep_asum(armas_d_dense_t *X, double start, int prec);
extern double ep_axpy(armas_d_dense_t *X, armas_d_dense_t *Y, double alpha, double beta, int prec);
extern void ep_gemv(armas_d_dense_t *Y, armas_d_dense_t *A, armas_d_dense_t *X,
                    double alpha, double beta, int prec, int flags);
extern void ep_gemm(armas_d_dense_t *C, armas_d_dense_t *A, armas_d_dense_t *B,
                    double alpha, double beta, int prec, int flags);

extern void ep_gendot(double *dot, double *tcond,
                      armas_d_dense_t *X, armas_d_dense_t *Y, double cond);
extern void ep_gensum(double *dot, double *tcond, armas_d_dense_t *X, double cond);
extern void ep_genmat(double *dot, double *tcond, armas_d_dense_t *A, armas_d_dense_t *B, double cond);
extern void ep_gentrm(double *dot, double *tcond, armas_d_dense_t *A, armas_d_dense_t *B, double cond, int flags);
#endif


// compute res + x.T*y 
__Dtype ep_dot_n(__Matrix *X, __Matrix *Y, __Dtype start, int N, int prec)
{
    int k;
    __Dtype de;
    mpfr_t x0, y0, s0, p0;

    if (prec <= 27) {
        prec = 200;
    }

    mpfr_init2(x0, prec);
    mpfr_init2(y0, prec);
    mpfr_init2(s0, prec);
    mpfr_init2(p0, prec);
    
    mpfr_set_d(s0, start, MPFR_RNDN);
    for (k = 0; k < N; k++) {
        mpfr_set_d(x0, (double)matrix_get_at(X, k), MPFR_RNDN);
        mpfr_set_d(y0, (double)matrix_get_at(Y, k), MPFR_RNDN);
        mpfr_mul(p0, x0, y0, MPFR_RNDN);
        mpfr_add(s0, s0, p0, MPFR_RNDN);
    }

    de = (__Dtype)mpfr_get_d(s0, MPFR_RNDN);
    mpfr_clear(x0);
    mpfr_clear(y0);
    mpfr_clear(s0);
    mpfr_clear(p0);
    return de;
}

__Dtype ep_dot(__Matrix *X, __Matrix *Y, __Dtype start, int prec)
{
    return ep_dot_n(X, Y, start, matrix_size(X), prec);
}

void ep_axpy(__Matrix *X, __Matrix *Y, __Dtype alpha, __Dtype beta, int prec)
{
    int k, N;
    __Dtype t;
    mpfr_t a0, b0, x0, y0, x1, y1;

    if (prec <= 27) {
        prec = 200;
    }
    N = matrix_size(X);

    mpfr_init2(x0, prec);
    mpfr_init2(y0, prec);
    mpfr_init2(x1, prec);
    mpfr_init2(y1, prec);
    mpfr_init2(a0, prec);
    mpfr_init2(b0, prec);
    
    mpfr_set_d(a0, (double)alpha, MPFR_RNDN);
    mpfr_set_d(b0, (double)beta, MPFR_RNDN);
    for (k = 0; k < N; k++) {
        mpfr_set_d(x0, (double)matrix_get_at(X, k), MPFR_RNDN);
        mpfr_set_d(y0, (double)matrix_get_at(Y, k), MPFR_RNDN);
        // x = alpha*x[k]
        mpfr_mul(x1, a0, x0, MPFR_RNDN);
        // y = beta*y[k]
        mpfr_mul(y1, b0, y0, MPFR_RNDN);
        // y = y + x
        mpfr_add(y0, y1, x1, MPFR_RNDN);
        t = (__Dtype)mpfr_get_d(y0, MPFR_RNDN);
        matrix_set_at(Y, k, t);
    }
    //mpfr_printf("%.20RNe\n", s0);
    mpfr_clear(x0);
    mpfr_clear(y0);
    mpfr_clear(x1);
    mpfr_clear(y1);
    mpfr_clear(a0);
    mpfr_clear(b0);
}

// compute: start + sqrt(x.T*x)
__Dtype ep_nrm2(__Matrix *X, __Dtype start, int prec)
{
    int k, N;
    __Dtype de;
    mpfr_t x0, s0, p0;

    N = matrix_size(X);

    if (prec <= 27) {
        prec = 200;
    }

    mpfr_init2(x0, prec);
    mpfr_init2(s0, prec);
    mpfr_init2(p0, prec);
    
    mpfr_set_d(s0, (double)start, MPFR_RNDN);
    for (k = 0; k < N; k++) {
        mpfr_set_d(x0, (double)matrix_get_at(X, k), MPFR_RNDN);
        mpfr_mul(p0, x0, x0, MPFR_RNDN);
        mpfr_add(s0, s0, p0, MPFR_RNDN);
    }

    mpfr_sqrt(x0, s0, MPFR_RNDN);
    de = (__Dtype)mpfr_get_d(x0, MPFR_RNDN);
    mpfr_clear(x0);
    mpfr_clear(s0);
    mpfr_clear(p0);
    return de;
}

// compute res + sum(x)
__Dtype ep_sum(__Matrix *X, __Dtype start, int prec)
{
    int k, N;
    __Dtype de;
    mpfr_t x0, s0;

    N = matrix_size(X);

    if (prec <= 27) {
        prec = 200;
    }

    mpfr_init2(x0, prec);
    mpfr_init2(s0, prec);
    
    mpfr_set_d(s0, start, MPFR_RNDN);
    for (k = 0; k < N; k++) {
        mpfr_set_d(x0, (double)matrix_get_at(X, k), MPFR_RNDN);
        mpfr_add(s0, s0, x0, MPFR_RNDN);
    }

    de = (__Dtype)mpfr_get_d(s0, MPFR_RNDN);
    mpfr_clear(x0);
    mpfr_clear(s0);
    return de;
}

__Dtype ep_asum(__Matrix *X, __Dtype start, int prec)
{
    int k, N;
    __Dtype de;
    mpfr_t x0, s0;

    N = matrix_size(X);
    if (prec <= 27) {
        prec = 200;
    }

    mpfr_init2(x0, prec);
    mpfr_init2(s0, prec);
    
    mpfr_set_d(s0, start, MPFR_RNDN);
    for (k = 0; k < N; k++) {
        mpfr_set_d(x0, fabs((double)matrix_get_at(X, k)), MPFR_RNDN);
        mpfr_add(s0, s0, x0, MPFR_RNDN);
    }
    de = (__Dtype)mpfr_get_d(s0, MPFR_RNDN);
    mpfr_clear(x0);
    mpfr_clear(s0);
    return de;
}

// y = beta*y + alpha*A*x; Y,X column vectors
void ep_gemv(__Matrix *Y, __Matrix *A, __Matrix *X,
             __Dtype alpha, __Dtype beta, int prec, int flags)
{
    int i, j, M, N, trans;
    __Dtype t;
    mpfr_t a0, b0, x0, y0, a1;

    trans = flags & ARMAS_TRANS ? 1 : 0;

    M = armas_d_size(Y);
    N = armas_d_size(X);

    if (prec <= 27) {
        prec = 200;
    }

    mpfr_init2(x0, prec);
    mpfr_init2(y0, prec);
    mpfr_init2(a1, prec);
    mpfr_init2(a0, prec);
    mpfr_init2(b0, prec);
    
    mpfr_set_d(a0, alpha, MPFR_RNDN);
    mpfr_set_d(b0, beta, MPFR_RNDN);

    for (i = 0; i < M; i++) {
        mpfr_set_d(y0, 0.0, MPFR_RNDN);
        for (j = 0; j < N; j++) {
            if (trans) {
                mpfr_set_d(a1, (double)matrix_get(A, j, i), MPFR_RNDN);
            } else {
                mpfr_set_d(a1, (double)matrix_get(A, i, j), MPFR_RNDN);
            }
            mpfr_set_d(x0, armas_d_get_at(X, j), MPFR_RNDN);
            mpfr_mul(x0, a1, x0, MPFR_RNDN);
            mpfr_add(y0, y0, x0, MPFR_RNDN);
        }
        mpfr_mul(y0, a0, y0, MPFR_RNDN);
        // here: y0 = alpha*(sum(A[i,:]*x[:])
        mpfr_set_d(x0, armas_d_get_at(Y, i), MPFR_RNDN);
        mpfr_mul(x0, b0, x0, MPFR_RNDN);
        // here: x0 = beta*y[i]
        mpfr_add(y0, x0, y0, MPFR_RNDN);
        t = (__Dtype)mpfr_get_d(y0, MPFR_RNDN);
        matrix_set_at(Y, i, t);
    }
    mpfr_clear(x0);
    mpfr_clear(y0);
    mpfr_clear(a1);
    mpfr_clear(a0);
    mpfr_clear(b0);
}

void ep_gemm(__Matrix *C, __Matrix *A, __Matrix *B,
             __Dtype alpha, __Dtype beta, int prec, int flags)
{
    __Matrix cv, bv;
    int i, trans;

    trans = flags & ARMAS_TRANSA ? ARMAS_TRANS : 0;

    for (i = 0; i < C->cols; i++) {
        matrix_column(&cv, C, i);
        if (flags & ARMAS_TRANSB) {
            matrix_row(&bv, B, i);
        } else {
            matrix_column(&bv, B, i);
        }
        ep_gemv(&cv, A, &bv, alpha, beta, prec, trans);
    }
}

static void seed()
{
    static int init = 0;
    if (!init) {
        srand48((long)time(0));
        init = 1;
    }
}


// simple permutation of vector; 
void permute(__Dtype *p, int N)
{
    int i, j;
    __Dtype t;

    seed();
    for (i = 0, j = N-1; i < j; i++, j--) {
        // swap p[i] and p[j] if low bit set.
        if (lrand48() & 0x1) {
            t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
    }
}

// simple permutation of two vectors; 
void permute2(__Dtype *p, __Dtype *r, int N)
{
    int i, j;
    __Dtype t;

    seed();
    for (i = 0, j = N-1; i < j; i++, j--) {
        // swap p[i] and p[j] if low bit set.
        if (lrand48() & 0x1) {
            t = p[i];
            p[i] = p[j];
            p[j] = t;
            t = r[i];
            r[i] = r[j];
            r[j] = t;
        }
    }
}

// simple permutation of vector; 
void vpermute(__Matrix *P)
{
    int i, j, N;
    __Dtype t;

    N = matrix_size(P);
    seed();
    for (i = 0, j = N-1; i < j; i++, j--) {
        // swap p[i] and p[j] if low bit set.
        if (lrand48() & 0x1) {
            t = matrix_get_at(P, i);
            matrix_set_at(P, i, matrix_get_at(P, j));
            matrix_set_at(P, j, t);
        }
    }
}
// simple permutation of two vectors; 
void vpermute2(__Matrix *P, __Matrix *R)
{
    int i, j, N;
    __Dtype t;

    N = matrix_size(P);
    seed();
    for (i = 0, j = N-1; i < j; i++, j--) {
        // swap p[i] and p[j] if low bit set.
        if (lrand48() & 0x1) {
            t = matrix_get_at(P, i);
            matrix_set_at(P, i, matrix_get_at(P, j));
            matrix_set_at(P, j, t);

            t = matrix_get_at(R, i);
            matrix_set_at(R, i, matrix_get_at(R, j));
            matrix_set_at(R, j, t);
        }
    }
}

#ifndef __MBITS
#define __MBITS 53
#endif

/*
 * \brief Generate ill-conditioned dot products
 *
 * \param[out] dot
 *      Dot product rounded to nearest
 * \param[out] tcond
 *      Actual condition number
 * \param[out] x, y
 *      Generated vectors
 * \param[in] N
 *      Vector length
 * \param[in] cond
 *      Anticipated condition number
 *
 * Algorithm 6.1 in (1)
 */
//void prec_gendot(double *dot, double *tcond, double *x, double *y, int N, double cond)
void ep_gendot(double *dot, double *tcond, __Matrix *X, __Matrix *Y, double cond)
{
    double b, e, de, s, p, h, q, x0, y0;
    int n2, i, N;

    N = armas_d_size(X);
    n2 = N/2 + (N&0x1);
    b = log2(cond);

    seed();

    // generate first half [0..n2-1] of vectors with random exponent
    e       = round(b/2.0);
    x0    = (2.0*drand48() - 1.0)*pow(2.0, e);
    y0    = (2.0*drand48() - 1.0)*pow(2.0, e);
    matrix_set_at(X, 0, (__Dtype)x0);
    matrix_set_at(Y, 0, (__Dtype)y0);

    for (i = 1; i < n2-1; i++) {
        e    = round(drand48()*(b/2.0));
        x0 = (2.0*drand48() - 1.0)*pow(2.0, e);
        y0 = (2.0*drand48() - 1.0)*pow(2.0, e);
        matrix_set_at(X, i, (__Dtype)x0);
        matrix_set_at(Y, i, (__Dtype)y0);
    }
    // make sure exponents b/2 and 0 occur
    x0 = (2.0*drand48() - 1.0);
    y0 = (2.0*drand48() - 1.0);
    matrix_set_at(X, n2-1, (__Dtype)x0);
    matrix_set_at(Y, n2-1, (__Dtype)y0);
    
    // second [n2..N] half with decreasing exponent
    for (i = n2; i < N; i++) {
        // e is decreasing linearilly from b/2 to 0
        e = round((b/2.0)*((double)(N-1-i)/(double)(N-1-n2)));
        x0 = (2.0*drand48() - 1.0)*pow(2.0, e);
        y0 = (2.0*drand48() - 1.0)*pow(2.0, e);
        matrix_set_at(X, i, (__Dtype)x0);
        matrix_set_at(Y, i, (__Dtype)y0);
    }

    // scale y in second half
    for (i = n2; i < N; i++) {
        // excat dot product; 4-fold precission
        de = ep_dot_n(X, Y, 0.0, i, 7*__MBITS);  
        //printf("%3d: %13e, %13e, %13e\n", i, y[i], de, x[i]);
        x0 = (double)matrix_get_at(X, i);
        y0 = (double)matrix_get_at(Y, i);
        matrix_set_at(Y, i, (__Dtype)(y0-de/x0));
    }
    
    // permute x and y
    vpermute2(X, Y);
    *dot = (double)ep_dot(X, Y, 0.0, 7*__MBITS);
    //sqde = sqrt(fabs(*dot));

    // compute condition number; C = 2*|x.T|*|y|/|x.T*y|
    *tcond = 0.0;
    s  = 0.0;
    for (i = 0; i < N; i++) {
        x0 = (double)matrix_get_at(X, i);
        y0 = (double)matrix_get_at(Y, i);
        twoprod_f64(&h, &q, fabs(x0), fabs(y0)/fabs(*dot));
        twosum_f64(tcond, &p, *tcond, h);
        s += p + q;
    }
    *tcond += s;
    *tcond *= 2.0;
}


// Generate ill-conditioned vector for sum; N must be even.
void ep_gensum(double *dot, double *tcond, armas_d_dense_t *X, double cond)
{
    double p, q;
    armas_d_dense_t X0, Y0;
    double x0, y0;
    int k;
    int N = armas_d_size(X);

    armas_d_make(&X0, N/2, 1, N/2, armas_d_data(X));
    armas_d_make(&Y0, N/2, 1, N/2, &armas_d_data(X)[N/2]);
    ep_gendot(dot, tcond, &X0, &Y0, cond);
    for (k = 0; k < N/2; k++) {
        x0 = armas_d_get_at(&X0, k);
        y0 = armas_d_get_at(&Y0, k);
        twoprod(&p, &q, x0, y0);
        armas_d_set_at(&X0, k, p);
        armas_d_set_at(&Y0, k, q);
    }
    vpermute(X);
}

void ep_genmat(double *dot, double *tcond, __Matrix *A, __Matrix *B, double cond)
{
    __Matrix R0, C0, R1, C1;
    int k;

    matrix_row(&R0, A, 0);
    matrix_column(&C0, B, 0);
    ep_gendot(dot, tcond, &R0, &C0, cond);

    // make rest of rows/columns permutations of first row/column.
    for (k = 1; k < A->rows; k++) {
        matrix_row(&R1, A, k);
        matrix_column(&C1, B, k);
        matrix_mcopy(&R1, &R0);
        matrix_mcopy(&C1, &C0);
        vpermute2(&R1, &C1);
    }
}

static double __zeros(int i, int j)
{
    return 0.0;
}

static double __ones(int i, int j)
{
    return 1.0;
}

void ep_gentrm(double *dot, double *tcond, __Matrix *A, __Matrix *B, double cond, int flags)
{
    __Matrix R0, C0, C1, D;
    int k;
    int right = flags & ARMAS_RIGHT;

    // make A identity
    matrix_set_values(A, __zeros, 0);
    matrix_diag(&D, A, 0);
    matrix_set_values(&D, __ones, 0);
    
    switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_UPPER|ARMAS_RIGHT:
    case ARMAS_UPPER|ARMAS_TRANS:
        matrix_column(&R0, A, A->cols-1);
        break;

    case ARMAS_LOWER|ARMAS_RIGHT:
    case ARMAS_LOWER|ARMAS_TRANS:
        matrix_column(&R0, A, 0);
        break;

    case ARMAS_LOWER|ARMAS_TRANS|ARMAS_RIGHT:
    case ARMAS_LOWER:
        matrix_row(&R0, A, A->rows-1);
        break;

    case ARMAS_UPPER|ARMAS_TRANS|ARMAS_RIGHT:
    case ARMAS_UPPER:
    default:
        matrix_row(&R0, A, 0);
        break;
    }

    if (right) {
        matrix_row(&C0, B, 0);
    } else {
        matrix_column(&C0, B, 0);
    }
    ep_gendot(dot, tcond, &R0, &C0, cond);

     // make rest of rows/columns copies of first row/column.
    for (k = 1; k < (right ? B->rows : B->cols); k++) {
        if (right) {
            matrix_row(&C1, B, k);
        } else {
            matrix_column(&C1, B, k);
        }
        matrix_mcopy(&C1, &C0);
    }
}


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:

