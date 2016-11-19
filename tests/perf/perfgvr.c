
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>

#include "unit/testing.h"

static inline
void gvright(armas_x_dense_t *A, double c, double s, int c1, int c2, int row, int nrow)
{
    double v0, v1, t0, t1, *y0, *y1;
    int k;

    y0 = &A->elems[c1*A->step+row];
    y1 = &A->elems[c2*A->step+row];
    for (k = 0; k < nrow; k++) {
        t0    = c * y0[k] + s * y1[k];
        y1[k] = c * y1[k] - s * y0[k];
        y0[k] = t0;
    }
}

int test_gvright(armas_x_dense_t *A, double c, double s)
{
  int k;
  for (k = 0; k < A->cols-1; k++) {
    gvright(A, c, s, k, k+1, 0, A->rows);
  }
  return 0;
}

#ifdef __AVX__
static inline
void gvright_avx(armas_x_dense_t *A, double c, double s, int c1, int c2, int row, int nrow)
{
    register __m256d V0, V1, V2, V3, T0, T1, X0, X1, COS, SIN;
    double *y0, *y1, t0;
    int k;
    
    y0 = &A->elems[c1*A->step+row];
    y1 = &A->elems[c2*A->step+row];
    COS = _mm256_set1_pd(c);
    SIN = _mm256_set1_pd(s);
    
    V0 = _mm256_loadu_pd(&y0[0]);
    V1 = _mm256_loadu_pd(&y1[0]);
    // t0 = c * v0 + s * v1;
    // t1 = c * v1 - s * v0;
    for (k = 0; k < nrow-3; k += 4) {
        V2 = _mm256_loadu_pd(&y0[k+4]);
        V3 = _mm256_loadu_pd(&y1[k+4]);
	T0 = COS*V0;
	T1 = COS*V1;
	X0 = T0 + V1*SIN;
	X1 = T1 - V0*SIN;
        _mm256_storeu_pd(&y0[k], X0);
        _mm256_storeu_pd(&y1[k], X1);
        V0 = V2; V1 = V3;
    }
    if (k == nrow)
        return;

    for (; k < nrow; k++) {
        t0    = c * y0[k] + s * y1[k];
        y1[k] = c * y1[k] - s * y0[k];
        y0[k] = t0;
    }
}
#endif


int test_gvright_avx(armas_x_dense_t *A, double c, double s)
{
#ifdef __AVX__
  int k;
  for (k = 0; k < A->cols-1; k++) {
    gvright_avx(A, c, s, k, k+1, 0, A->rows);
  }
  return 0;
#else
  printf("AVX instruction set not supported.\n");
  return 0;
#endif
}

// ------------------------------------------------------------------------------

static inline
void gvleft(armas_x_dense_t *A, double c, double s, int r1, int r2, int col, int ncol)
{
    double t0, t1, *y0, *y1;
    int k, n;

    y0 = &A->elems[col*A->step+r1];
    y1 = &A->elems[col*A->step+r2];
    for (k = 0, n = 0; k < ncol; k++, n += A->step) {
        t0    = c * y0[n] + s * y1[n];
        y1[n] = c * y1[n] - s * y0[n];
        y0[n] = t0;
    }
}

int test_gvleft(armas_x_dense_t *A, double c, double s)
{
  int k;
  for (k = 0; k < A->rows-1; k++) {
    gvleft(A, c, s, k, k+1, 0, A->cols);
  }
  return 0;
}

#ifdef __AVX__
void gvleft_avx(armas_x_dense_t *A, double c, double s, int r1, int r2, int col, int ncol)
{
    register __m256d V0, V1, V2, V3, T0, T1, T2, T3, X0, X1, X2, X3, COS, SIN, Z;
    double *y0, *y1, t0;
    int k, n0, n1, n2, n3, n4, n5, n6, n7, nshift;
    
    y0 = &A->elems[col*A->step+r1];
    y1 = &A->elems[col*A->step+r2];
    COS = _mm256_set1_pd(c);
    SIN = _mm256_set1_pd(s);
    
    // t0 = c * v0 + s * v1;
    // t1 = c * v1 - s * v0;
    n0 = 0;            n1 = n0 + A->step; n2 = n1 + A->step; n3 = n2 + A->step;
    n4 = n3 + A->step; n5 = n4 + A->step; n6 = n5 + A->step; n7 = n6 + A->step;
    V0 = _mm256_set_pd(y0[n3], y0[n2], y0[n1], y0[n0]);
    V1 = _mm256_set_pd(y1[n3], y1[n2], y1[n1], y1[n0]);

    nshift = A->step << 2;
    for (k = 0; k < ncol-3; k += 4) {
        V2 = _mm256_set_pd(y0[n7], y0[n6], y0[n5], y0[n4]);
        V3 = _mm256_set_pd(y1[n7], y1[n6], y1[n5], y1[n4]);
	T0 = COS*V0;
	T1 = COS*V1;
	X0 = T0 + V1*SIN;
	X1 = T1 - V0*SIN;
        y0[n3] = X0[3]; y0[n2] = X0[2]; y0[n1] = X0[1]; y0[n0] = X0[0];
        y1[n3] = X1[3]; y1[n2] = X1[2]; y1[n1] = X1[1]; y1[n0] = X1[0];

        n3 += nshift; n2 += nshift; n1 += nshift; n0 += nshift;
        n4 += nshift; n5 += nshift; n6 += nshift; n7 += nshift;
        V0 = V2; V1 = V3;
    }
    if (k == ncol)
        return;

    for (; k < ncol; k++, n0 += A->step) {
        t0    = c * y0[n0] + s * y1[n0];
        y1[n0] = c * y1[n0] - s * y0[n0];
        y0[n0] = t0;
    }
}
#endif

int test_gvleft_avx(armas_x_dense_t *A, double c, double s)
{
#ifdef __AVX__
  int k;
  for (k = 0; k < A->rows-1; k++) {
    gvleft_avx(A, c, s, k, k+1, 0, A->cols);
  }
  return 0;
#else
  printf("AVX instruction set not supported.\n");
  return 0;
#endif
}

// ------------------------------------------------------------------------------

int main(int argc, char **argv)
{

  int verbose = 0;
  int count = 5;
  double rt, min, max, avg;
  double a = 0.7, b = -0.88, c;
  armas_x_dense_t A;

  int ok, opt, i;
  long k, N = 3000, M = 10;
  int nproc = 1;
  int avx_test = 0;
  int left = 0;
  int testno = 0;

  while ((opt = getopt(argc, argv, "c:vt:AL")) != -1) {
    switch (opt) {
    case 'v':
      verbose = 1;
      break;
    case 't':
      testno = atoi(optarg);
      break;
    case 'c':
      count = atoi(optarg);
      break;
    case 'A':
      avx_test = 1;
      break;
    case 'L':
      left = 1;
      break;
    default:
      fprintf(stderr, "usage: perfgv [-c count -v] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atol(argv[optind]);

  long seed = (long)time(0);
  srand48(seed);

  armas_x_init(&A, N, N);
  armas_x_set_values(&A, unitrand, ARMAS_ANY);

  // C = A*B
  min = max = avg = 0.0;
  for (i = 0; i < count; i++) {
    flush();

    if (left) {
        if (avx_test) {
            rt = time_msec();
            c = test_gvleft_avx(&A, -0.6, -0.5);
            rt = time_msec() - rt;
        } else {
            rt = time_msec();
            c = test_gvleft(&A, -0.6, -0.5);
            rt = time_msec() - rt;
        }
    } else {
        if (avx_test) {
            rt = time_msec();
            c = test_gvright_avx(&A, -0.6, -0.5);
            rt = time_msec() - rt;
        } else {
            rt = time_msec();
            c = test_gvright(&A, -0.6, -0.5);
            rt = time_msec() - rt;
        }
    }


    if (i == 0) {
      min = max = avg = rt;
    } else {
      if (rt < min)
	min = rt;
      if (rt > max)
	max = rt;
      avg += (rt - avg)/(i+1);
    }
    if (verbose)
      printf("%2d: %.4f, %.4f, %.4f msec\n", i, min, avg, max);
  }

  int64_t nops = 6*(int64_t)N*N;

  printf("N: %4ld, %8.4f, %8.4f, %8.4f Gflops\n", N,
         gflops(max, nops), gflops(avg, nops), gflops(min, nops));
  printf("c=%.f\n", c);
  return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
