
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>

#include "unit/testing.h"

//---------------------------------------------------------------------------------

void gvrotg(double *c, double *s, double *r, double a, double b)
{
    double t, u;

    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;
    } else if (a == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = b;
    } else if (fabs(b) > fabs(a)) {
        t = a / b;
        u = sqrt(1.0 + t*t);
        if (signbit(b))
            u = -u;
        *s = 1.0 / u;
        *c = (*s) * t;
        *r = b * u;
    } else {
        t = b / a;
        u = sqrt(1.0 + t*t);
        *r = a * u;
        *c = 1.0 / u;
        *s = (*c) * t;
    }
}

#ifdef __AVX__
void gvrotg_avx(double *c, double *s, double *r, double a, double b)
{
    register __m256d x0, x1, t0, t2, u0, u1, one, b0, b1;
    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;
        return;
    }
    if (a == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = b;
        return;
    }

    // set_pd() order: [3, 2, 1, 0]
    // x[0], x[1]: |a| > |b|,  x[2],x[3]: |b| > |a|

    x0  = _mm256_set_pd(1.0, a, b, 1.0);   // x0 = {1, a,   b,   1}
    x1  = _mm256_set_pd(1.0, b, a, 1.0);   // x0 = {1, b,   a,   1}
    t0  = _mm256_div_pd(x0, x1);           // t0 = {1, a/b, b/a, 1}
    x0  = _mm256_mul_pd(t0, t0);           // x3 = {1, (a/b)^2, (b/a)^2, 1}
    t2  = _mm256_hadd_pd(x0, x0);          // x3 = {1+(a/b)^2, ., (b/a)^2+1, ..}
    u0  = _mm256_sqrt_pd(t2);              // u0 = {sqrt(1+(a/b)^2), .., sqrt((b/a)^2+1)}
    one = _mm256_set1_pd(1.0);
    u1  = _mm256_div_pd(one, u0);
    b0  = _mm256_blend_pd(u0, u1, 0x9);    // b0 = {1/u(b),   u(b),   u(a), 1/u(a)} 
    b0  = _mm256_mul_pd(b0, x1);           // b0 = {1/u(b), b*u(b), a*u(a), 1/u(a)} 
    b1  = _mm256_mul_pd(t0, u1);           // b1 = {1/u(b), t*u(b), t*u(a), 1/u(a)} 

    if (fabs(b) > fabs(a)) {
      *s = b0[3];  // = 1/u(b)
      *r = b0[2];  // = b*u(b)
      *c = b1[2];  // = t*u(b)
      if (signbit(b)) {
          *s = -(*s);
          *c = -(*c);
          *r = -(*r);
      }
    } else {
      *c = b0[0];
      *r = b0[1];
      *s = b1[1];
    }
}
#endif

void gvrotg_fma(double *c, double *s, double *r, double a, double b)
{
#if defined(__FMA__)
    register __m256d x0, x1, t0, t2, u0, u1, one, b0, b1;
    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;
        return;
    }
    if (a == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = b;
        return;
    }

    // set_pd() order: [3, 2, 1, 0]
    // x[0], x[1]: |a| > |b|,  x[2],x[3]: |b| > |a|

    one = _mm256_set1_pd(1.0);
    x0  = _mm256_set_pd(1.0, a, b, 1.0);   // x0 = {1, a,   b,   1}
    x1  = _mm256_set_pd(1.0, b, a, 1.0);   // x0 = {1, b,   a,   1}
    t0  = _mm256_div_pd(x0, x1);           // t0 = {1, a/b, b/a, 1}
    t2  = _mm256_fmadd_pd(t0, t0, one);    // x3 = {1, 1+(a/b)^2, (b/a)^2+1, 1}
    u0  = _mm256_sqrt_pd(t2);              // u0 = {1, sqrt(1+(a/b)^2), sqrt((b/2)^2+1), 1}
    u1  = _mm256_div_pd(one, u0);
    b0  = _mm256_blend_pd(u0, u1, 0x9);    // b0 = {1/u(a),   u(a),   u(b), 1/u(b)} 
    b0  = _mm256_mul_pd(b0, x1);           // b0 = {1/u(a), b*u(a), a*u(b), 1/u(b)} 
    b1  = _mm256_mul_pd(t0, u1);           // b1 = {1/u(a), t*u(a), t*u(b), 1/u(b)} 

    if (fabs(b) > fabs(a)) {
      *s = b0[3];
      *r = b0[2];
      *c = b1[2];
      if (signbit(b)) {
          *s = -(*s);
          *c = -(*c);
          *r = -(*r);
      }
    } else {
      *c = b0[0];
      *r = b0[1];
      *s = b1[1];
    }
#endif
}

int test_gvcomp_avx(double a, double b, long N)
{
#if defined(__AVX__)
  double c, s, r, t0;
  long k;

  for (k = 0; k < N; k++) {
    gvrotg_avx(&c, &s, &r, a, b);
  }
  return (c > 0.0 && s > 0.0 && r > 0.0) ? 1 : 0;
#else
  printf("AVX instruction set not supported...\n");
  return 0;
#endif
}

int test_gvcomp_fma(double a, double b, long N)
{
#if defined(__FMA__)
  double c, s, r;
  long k;

  for (k = 0; k < N; k++) {
    gvrotg_fma(&c, &s, &r, a, b);
  }
  return (c > 0.0 && s > 0.0 && r > 0.0) ? 1 : 0;
#else
  printf("FMA instruction set not supported...\n");
  return 0;
#endif
}

int test_gvcomp(double a, double b, long N)
{
  double c, s, r;
  long k;

  for (k = 0; k < N; k++) {
    gvrotg(&c, &s, &r, a, b);
  }
  return (c > 0.0 && s > 0.0 && r > 0.0) ? 1 : 0;
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
  long k, N = 30000000;
  int nproc = 1;
  int avx_test = 0;
  int fma_test = 0;
  int testno = 0;

  while ((opt = getopt(argc, argv, "c:vt:AF")) != -1) {
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
    case 'F':
      fma_test = 1;
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

  // C = A*B
  min = max = avg = 0.0;
  for (i = 0; i < count; i++) {
    flush();

    if (fma_test) {
      rt = time_msec();
      c = test_gvcomp_fma(a, b, N);
      rt = time_msec() - rt;
    } else if (avx_test) {
      rt = time_msec();
      c = test_gvcomp_avx(a, b, N);
      rt = time_msec() - rt;
    } else {
      rt = time_msec();
      c = test_gvcomp(a, b, N);
      rt = time_msec() - rt;
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

  int64_t nops = 7*(int64_t)N;
  if (testno == 0) {
    nops = N;
    nops *= 7;
  } else if (testno == 2) {
    nops = N*N;
    nops *= 7;
  }
  printf("N: %4ld, %8.4f, %8.4f, %8.4f Gflops\n", N,
         gflops(max, nops), gflops(avg, nops), gflops(min, nops));
  printf("c=%.f\n", c);
  return 0;
}

// Local Variables:
// indent-tabs-mode: nil
// End:
