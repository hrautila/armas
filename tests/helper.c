
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

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
void flush() {
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
		    min(9, A->rows-row), min(9, A->cols-col), -1);
  printf("A, [%d, %d]\n", row, col); armas_d_printf(stdout, "%9.2e", &S);

  return ok;
}


