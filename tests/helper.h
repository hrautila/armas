
#ifndef __TEST_HELPER_H
#define __TEST_HELPER_H

#include <math.h>

extern double zero(int i, int j);
extern double one(int i, int j);
extern double zeromean(int i, int j);
extern double unitrand(int i, int j);
extern double stdrand(int i, int j);
extern double rowno(int i, int j);
extern double colno(int i, int j);

extern double time_msec();
extern double gflops(double ms, int64_t count);  
extern void flush();
extern int isOK(double nrm, int N);
extern int isFINE(double nrm, double tol);
extern double rel_error(double *dnorm, armas_d_dense_t *computed,
			armas_d_dense_t *expected, int norm, int flags, armas_conf_t *conf);

extern int check(armas_d_dense_t *A, armas_d_dense_t *B, int chkdir, const char *msg);
extern armas_d_dense_t *col_as_row(armas_d_dense_t *row, armas_d_dense_t *col);
extern void matrix_printf(FILE *out, const char *efmt, const armas_d_dense_t *m, int partial);

#define PASS(exp) ((exp) ? "PASS" : "FAIL")

static inline int ndigits(double err)
{
  return err != 0.0 ? (int)(ceil(log10(1.0/err))) : -1;
}
static inline int imin(int a, int b)
{
  return a < b ? a : b;
}

static inline int imax(int a, int b)
{
  return a > b ? a : b;
}


#endif
