
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef float __Dtype;
typedef armas_s_dense_t __Matrix;
#define EPSILON FLT_EPSILON
#define FABS fabsf
#define matrix_is_vector    armas_s_isvector
#define matrix_axpy         armas_s_axpy
#define matrix_scale_plus armas_s_scale_plus
#define matrix_mnorm        armas_s_mnorm
#define matrix_make         armas_s_make
#define matrix_size         armas_s_size
#define matrix_data         armas_s_data

#else
#include <armas/dmatrix.h>
typedef double __Dtype;
typedef armas_d_dense_t __Matrix;

#define EPSILON DBL_EPSILON
#define FABS fabs
#define matrix_is_vector    armas_d_isvector
#define matrix_axpy         armas_d_axpy
#define matrix_scale_plus armas_d_scale_plus
#define matrix_mnorm        armas_d_mnorm
#define matrix_make         armas_d_make
#define matrix_size         armas_d_size
#define matrix_data         armas_d_data

#endif

static inline int max(int a, int b)
{
  return a > b ? a : b;
}

static inline int min(int a, int b)
{
  return a < b ? a : b;
}

#define FLUSH_SIZE 2048*2048
double flush() {
  static double *data = (double *)0;
  int i;
  double t = 0.0;
  if (!data) {
    data = (double *)malloc(FLUSH_SIZE*sizeof(double));
  }

  for (i = 0; i < FLUSH_SIZE; i++) {
    data[i] = 1e-12;
  }
  for (i = 0; i < FLUSH_SIZE; i++) {
    t += data[i];
  }
  if (t < 10.0) {
    t = 0.0;
  }
  return t;
}

double time_msec() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (1e+6*tv.tv_sec + tv.tv_usec)/1000.0;
}

double gflops(double ms, int64_t count)
{
  return (count/ms)*1e-6;
}

__Dtype zero(int i, int j) {
  return 0.0;
}

__Dtype one(int i, int j) {
  return 1.0;
}

__Dtype rowno(int i, int j) {
  return (i+1)*1.0;
}

__Dtype colno(int i, int j) {
  return (j+1)*1.0;
}

__Dtype zeromean(int i, int j) {
  static int init = 0;
  if (!init) {
    srand48((long)time(0));
    init = 1;
  }
  return (__Dtype)(drand48() - 0.5);
}

__Dtype unitrand(int i, int j) {
  static int init = 0;
  if (!init) {
    srand48((long)time(0));
    init = 1;
  }
  return (__Dtype)drand48();
}

// std random variable from unit random with box-muller transformation
__Dtype stdrand(int i, int j)
{ 
  double s, u, v;
  static int init = 0;
  if (!init) {
    srand48((long)time(0));
    init = 1;
  }
  do {
    u = 2.0*drand48() - 1.0;
    v = 2.0*drand48() - 1.0;
    s = u*u + v*v;
  } while (s == 0.0 || s >= 1.0);
  return (__Dtype)(u * sqrt(-2.0*log(s)/s));
}

int isOK(__Dtype nrm, int N)
{
  //int nk = (int64_t)(FABS(nrm/EPSILON));
  //return nrm != 0.0 && nk < (int64_t)N;
  // if exactly zero then something suspect
  return nrm != 0.0 && FABS(nrm) < (__Dtype)N*EPSILON;
}

int isFINE(__Dtype nrm, __Dtype tol)
{
  // if exactly zero then something suspect
  return nrm != 0.0 && FABS(nrm) < FABS(tol);
}

int in_tolerance(double a, double b)
{
  static const double RTOL = 1.0000000000000001e-05;
  static const double ATOL = 1e-8;

  double df = fabs(a - b);
  double ref = ATOL + RTOL * fabs(b);
  if (df > ref) {
    printf("a=%.7f b=%.7f, df= %.3e, ref=%.3e\n", a, b, df, ref);
  }
  return df <= ref;
}

// Compute relative error: ||computed - expected||/||expected||
// Returns norm ||computed - exptected|| in dnorm. 
// Note: contents of computed is destroyed.
__Dtype rel_error(__Dtype *dnorm, __Matrix *computed,
		  __Matrix *expected, int norm, int flags, armas_conf_t *conf)
{
    double cnrm, enrm;
    if (matrix_is_vector(computed)) {
      matrix_axpy(computed, expected, -1.0, conf);
    } else {
      // computed = computed - expected
      matrix_scale_plus(computed, expected, 1.0, -1.0, flags, conf);
    }
    // ||computed - expected||
    cnrm = matrix_mnorm(computed, norm, conf);
    if (dnorm)
      *dnorm = cnrm;
    // ||expected||
    enrm = matrix_mnorm(expected, norm, conf);
    return cnrm/enrm;
}

__Matrix *col_as_row(__Matrix *row, __Matrix *col)
{
  matrix_make(row, 1, matrix_size(col), 1, matrix_data(col));
  return row;
}

#if !defined(FLOAT32)
// search from top-left
void find_from_bottom_right(armas_d_dense_t *a, armas_d_dense_t *b, int *row, int *col)
{
  int k;
  armas_d_dense_t aC, bC;

  for (k = a->rows-1; k >= 0; k--) {
    armas_d_row(&aC, a, k);
    armas_d_row(&bC, b, k);
    if (! armas_d_allclose(&aC, &bC)) {
      *row = k;
      break;
    }
  }
  if (*row == -1)
    return;

  for (k = 0; k < a->rows; k++) {
    double a = armas_d_get(&aC, 0, k);
    double b = armas_d_get(&bC, 0, k);
    if ( ! in_tolerance(a, b)) {
      *col = k;
      break;
    }
  }
}

void find_from_top_left(armas_d_dense_t *a, armas_d_dense_t *b, int *row, int *col)
{
  int k;
  armas_d_dense_t aC, bC;

  for (k = 0; k < a->cols; k++) {
    armas_d_column(&aC, a, k);
    armas_d_column(&bC, b, k);
    if (! armas_d_allclose(&aC, &bC)) {
      *col = k;
      break;
    }
  }
  if (*col == -1)
    return;

  // search row on column
  for (k = 0; k < a->rows; k++) {
    double a = armas_d_get(&aC, k, 0);
    double b = armas_d_get(&bC, k, 0);
    if ( ! in_tolerance(a, b)) {
      *row = k;
      break;
    }
  }
}

int check(armas_d_dense_t *A, armas_d_dense_t *B, int chkdir, const char *msg)
{
  int ok = armas_d_allclose(A, B);
  printf("%6s: %s\n", ok ? "OK" : "FAILED", msg);
  if (ok)
    return ok;

  armas_d_dense_t S;
  int row = -1, col = -1;
  if (chkdir > 0) {
    // from top-left to bottom-right
    find_from_top_left(A, B, &row, &col);
  } else {
    // from bottom-right to top-left
    find_from_bottom_right(A, B, &row, &col);
  }
  printf("1: first difference at [%d, %d]\n", row, col);
  armas_d_submatrix(&S, A, row, col,
		    min(9, A->rows-row), min(9, A->cols-col));
  printf("A, [%d, %d]\n", row, col); armas_d_printf(stdout, "%9.2e", &S);

  return ok;
}

void matrix_printf(FILE *out, const char *efmt, const armas_d_dense_t *m, int partial)
{
  int i, j;
  if (!m)
    return;
  if (!efmt)
    efmt = "%8.1e";

  int rowpartial = partial && m->rows > 18 ;
  int colpartial = partial && m->cols > 9;
  for (i = 0; i < m->rows; i++ ) {
    printf("[");
    for (j = 0; j < m->cols; j++ ) {
      if (j > 0) {
	printf(", ");
      }
      printf(efmt, m->elems[j*m->step+i]);
      if (colpartial && j == 3) {
        j = m->cols - 5;
        printf(", ...");
      }
    }
    printf("]\n");
    if (rowpartial && i == 8) {
      printf(" ....\n");
      i = m->rows - 10;
    }
  }
}
#endif




