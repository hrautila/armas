
#ifndef __TEST_HELPER_H
#define __TEST_HELPER_H

#include <math.h>

// NOTE: this file is included __ONLY__ after __Dtype and __Matrix typedefs.

extern __Dtype zero(int i, int j);
extern __Dtype one(int i, int j);
extern __Dtype zeromean(int i, int j);
extern __Dtype unitrand(int i, int j);
extern __Dtype stdrand(int i, int j);
extern __Dtype rowno(int i, int j);
extern __Dtype colno(int i, int j);

extern __Dtype rel_error(__Dtype *dnorm, __Matrix *computed,
			__Matrix *expected, int norm, int flags, armas_conf_t *conf);

extern int check(__Matrix *A, __Matrix *B, int chkdir, const char *msg);
extern __Matrix *col_as_row(__Matrix *row, __Matrix *col);
extern void matrix_printf_partial(FILE *out, const char *efmt, const __Matrix *m, int partial);


extern int isOK(__Dtype nrm, int N);
extern int isFINE(__Dtype nrm, __Dtype tol);


extern double time_msec();
extern double gflops(double ms, int64_t count);  
extern void flush();

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
