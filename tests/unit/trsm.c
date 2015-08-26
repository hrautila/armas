
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t __Matrix ;
typedef float __Dtype;

#define matrix_init       armas_s_init
#define matrix_set_values armas_s_set_values
#define matrix_mult       armas_s_mult
#define matrix_mult_trm   armas_s_mult_trm
#define matrix_solve_trm  armas_s_solve_trm
#define matrix_transpose  armas_s_transpose
#define matrix_release    armas_s_release
#define matrix_mcopy      armas_s_mcopy
#define matrix_printf     armas_s_printf
#define __SCALING (__Dtype)((1 << 14) + 1)

#define STRTOF(arg)  strtof(arg, (char **)0);

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define matrix_init       armas_d_init
#define matrix_set_values armas_d_set_values
#define matrix_mult       armas_d_mult
#define matrix_mult_trm   armas_d_mult_trm
#define matrix_solve_trm  armas_d_solve_trm
#define matrix_transpose  armas_d_transpose
#define matrix_release    armas_d_release
#define matrix_mcopy      armas_d_mcopy
#define matrix_printf     armas_d_printf

#define __SCALING (__Dtype)((1 << 27) + 1)
#define STRTOF(arg)  strtod(arg, (char **)0);

#endif
#include "helper.h"

__Dtype scaledrand(int i, int j)
{
  __Dtype val = unitrand(i, j);
  return val*__SCALING;
}

static __Dtype Aconstant = 1.0;

__Dtype constant(int i, int j)
{
  return Aconstant;
}

int main(int argc, char **argv)
{
  armas_conf_t conf;
  __Matrix B0, A, B;

  int ok, opt;
  int N = 601;
  int verbose = 1;
  int fails = 0;
  __Dtype alpha = 1.0;
  __Dtype n0, n1;

  while ((opt = getopt(argc, argv, "vC:")) != -1) {
    switch (opt) {
    case 'v':
      verbose++;
      break;
    case 'C':
      Aconstant = STRTOF(optarg);
      break;
    default:
      fprintf(stderr, "usage: trsm [-P nproc] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf = *armas_conf_default();

  matrix_init(&A, N, N);
  matrix_init(&B, N, N);
  matrix_init(&B0, N, N);
  
  // Upper triangular matrix
  matrix_set_values(&A, one, ARMAS_UPPER);
  if (N < 10) {
    printf("A:\n"); matrix_printf(stdout, "%8.1e", &A);
  }
    

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  if (N < 10) {
    printf("A*B:\n"); matrix_printf(stdout, "%8.1e", &B);
  }
  matrix_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  if (N < 10) {
    printf("A.-1*B:\n"); matrix_printf(stdout, "%8.1e", &B);
  }
  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, L|U|N), A, L|U|N)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, R|U|N), A, R|U|N)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, L|U|T), A, L|U|T)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, R|U|T), A, R|U|T)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // Lower triangular matrix
  matrix_set_values(&A, one, ARMAS_LOWER);

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, L|L|N), A, L|L|N)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, R|L|N), A, R|L|N)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT|ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, L|L|T), A, L|L|T)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);
  matrix_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  matrix_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: B = solve_trm(mult_trm(B, A, R|L|T), A, R|L|T)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
