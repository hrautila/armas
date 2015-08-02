
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>

#include <armas/dmatrix.h>


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

double zero(int i, int j) {
  return 0.0;
}

double one(int i, int j) {
  return 1.0;
}

double rowno(int i, int j) {
  return (i+1)*1.0;
}

double colno(int i, int j) {
  return (j+1)*1.0;
}

double zeromean(int i, int j) {
  static int init = 0;
  if (!init) {
    srand48((long)time(0));
    init = 1;
  }
  return drand48() - 0.5;
}

double unitrand(int i, int j) {
  static int init = 0;
  if (!init) {
    srand48((long)time(0));
    init = 1;
  }
  return drand48();
}

// std random variable from unit random with box-muller transformation
double stdrand(int i, int j)
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
  return u * sqrt(-2.0*log(s)/s);
}

int isOK(double nrm, int N)
{
  int nk = (int64_t)(fabs(nrm/DBL_EPSILON));
  // if exactly zero then something suspect
  return nrm != 0.0 && nk < (int64_t)N;
}

int isFINE(double nrm, double tol)
{
  // if exactly zero then something suspect
  return nrm != 0.0 && fabs(nrm) < fabs(tol);
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
double rel_error(double *dnorm, armas_d_dense_t *computed,
		 armas_d_dense_t *expected, int norm, int flags, armas_conf_t *conf)
{
    double cnrm, enrm;
    if (armas_d_isvector(computed)) {
      armas_d_axpy(computed, expected, -1.0, conf);
    } else {
      // computed = computed - expected
      armas_d_scale_plus(computed, expected, 1.0, -1.0, flags, conf);
    }
    // ||computed - expected||
    cnrm = armas_d_mnorm(computed, norm, conf);
    if (dnorm)
      *dnorm = cnrm;
    // ||expected||
    enrm = armas_d_mnorm(expected, norm, conf);
    return cnrm/enrm;
}

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

armas_d_dense_t *col_as_row(armas_d_dense_t *row, armas_d_dense_t *col)
{
  armas_d_make(row, 1, armas_d_size(col), 1, armas_d_data(col));
  return row;
}



