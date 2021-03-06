
#ifndef __TEST_HELPER_H
#define __TEST_HELPER_H

#include <math.h>

// NOTE: this file is included __ONLY__ after DTYPE and armas_dense_t typedefs.

extern DTYPE zero(int i, int j);
extern DTYPE one(int i, int j);
extern DTYPE zeromean(int i, int j);
extern DTYPE unitrand(int i, int j);
extern DTYPE stdrand(int i, int j);
extern DTYPE rowno(int i, int j);
extern DTYPE colno(int i, int j);

extern DTYPE rel_error(DTYPE * dnorm, armas_dense_t * computed,
                       armas_dense_t * expected, int norm, int flags,
                       armas_conf_t * conf);

extern int check(armas_dense_t * A, armas_dense_t * B, int chkdir,
                 const char *msg);
extern armas_dense_t *col_as_row(armas_dense_t * row,
                                   armas_dense_t * col);
extern void armas_printf_partial(FILE * out, const char *efmt,
                                   const armas_dense_t * m, int partial);

extern void json_read_write(armas_dense_t **Aptr, armas_dense_t *A, int flags);

extern int isOK(DTYPE nrm, int N);
extern int isFINE(DTYPE nrm, DTYPE tol);


extern double time_msec();
extern double gflops(double ms, int64_t count);
extern void flush();

#define PASS(exp) ((exp) ? "PASS" : "FAIL")

static inline int ndigits(double err)
{
    return err != 0.0 ? (int) (ceil(log10(1.0 / err))) : -1;
}

static inline int imin(int a, int b)
{
    return a < b ? a : b;
}

static inline int imax(int a, int b)
{
    return a > b ? a : b;
}

// extended precision helpers.

extern DTYPE ep_dot_n(armas_dense_t * X, armas_dense_t * Y, DTYPE start,
                      int N, int prec);
extern DTYPE ep_dot(armas_dense_t * X, armas_dense_t * Y, DTYPE start,
                    int prec);
extern DTYPE ep_nrm2(armas_dense_t * X, DTYPE start, int prec);
extern DTYPE ep_sum(armas_dense_t * X, DTYPE start, int prec);
extern DTYPE ep_asum(armas_dense_t * X, DTYPE start, int prec);
extern void ep_axpy(armas_dense_t * X, armas_dense_t * Y, DTYPE alpha,
                    DTYPE beta, int prec);
extern void ep_gemv(armas_dense_t * Y, armas_dense_t * A,
                    armas_dense_t * X, DTYPE alpha, DTYPE beta, int prec,
                    int flags);
extern void ep_gemm(armas_dense_t * C, armas_dense_t * A,
                    armas_dense_t * B, DTYPE alpha, DTYPE beta, int prec,
                    int flags);

extern void ep_gendot(double *dot, double *tcond,
                      armas_dense_t * X, armas_dense_t * Y, double cond);
extern void ep_gensum(double *dot, double *tcond, armas_dense_t * X,
                      double cond);
extern void ep_genmat(double *dot, double *tcond, armas_dense_t * A,
                      armas_dense_t * B, double cond);
extern void ep_gentrm(double *dot, double *tcond, armas_dense_t * A,
                      armas_dense_t * B, double cond, int flags);

#endif
