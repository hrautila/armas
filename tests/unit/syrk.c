
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t __Matrix ;
typedef float __Dtype;

#define matrix_init       armas_s_init
#define matrix_set_values armas_s_set_values
#define matrix_mult       armas_s_mult
#define matrix_mult_trm   armas_s_mult_trm
#define matrix_make_trm   armas_s_make_trm
#define matrix_update_sym armas_s_update_sym
#define matrix_transpose  armas_s_transpose
#define matrix_release    armas_s_release
#define matrix_mcopy      armas_s_mcopy
#define matrix_printf     armas_s_printf
#define __SCALING (__Dtype)((1 << 14) + 1)

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define matrix_init       armas_d_init
#define matrix_set_values armas_d_set_values
#define matrix_mult       armas_d_mult
#define matrix_make_trm   armas_d_make_trm
#define matrix_mult_trm   armas_d_mult_trm
#define matrix_update_sym armas_d_update_sym
#define matrix_transpose  armas_d_transpose
#define matrix_release    armas_d_release
#define matrix_mcopy      armas_d_mcopy
#define matrix_printf     armas_d_printf

#define __SCALING (__Dtype)((1 << 27) + 1)

#endif
#include "helper.h"

int main(int argc, char **argv)
{
  armas_conf_t conf;
  __Matrix C, C0, A, At;

  int ok, opt;
  int N = 255;
  int verbose = 1, fails = 0;
  __Dtype n0, n1, alpha = 1.0;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose++;
      break;
    default:
      fprintf(stderr, "usage: xsyrk [-v] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf = *armas_conf_default();

  matrix_init(&C,  N, N);
  matrix_init(&C0, N, N);
  matrix_init(&A,  N, N/2);
  matrix_init(&At, N/2, N);
  
  matrix_set_values(&A, zeromean, ARMAS_NULL);
  matrix_transpose(&At, &A);

  // 1. C = upper(C) + A*A.T;
  matrix_set_values(&C, one, ARMAS_SYMM);
  matrix_mcopy(&C0, &C);

  matrix_make_trm(&C, ARMAS_UPPER);
  matrix_update_sym(&C, &A, alpha, 0.0, ARMAS_UPPER, &conf);

  matrix_mult(&C0, &A, &At, alpha, 0.0, ARMAS_NULL, &conf);
  matrix_make_trm(&C0, ARMAS_UPPER);
  //ok = matrix_allclose(&C0, &C);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;

  printf("%6s: syrk(C, A, U|N) == TriU(gemm(C, A, A.T))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // 2. C = upper(C) + A.T*A
  matrix_set_values(&C, one, ARMAS_SYMM);
  matrix_mcopy(&C0, &C);

  matrix_make_trm(&C, ARMAS_UPPER);
  matrix_update_sym(&C, &At, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);

  matrix_mult(&C0, &At, &At, alpha, 0.0, ARMAS_TRANSA, &conf);
  matrix_make_trm(&C0, ARMAS_UPPER);
  //ok = matrix_allclose(&C0, &C);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;

  printf("%6s: syrk(C, A, U|T) == TriU(gemm(C, A.T, A))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // 1. C = lower(C) + A*A.T;
  matrix_set_values(&C, one, ARMAS_SYMM);
  matrix_mcopy(&C0, &C);

  matrix_make_trm(&C, ARMAS_LOWER);
  matrix_update_sym(&C, &A, alpha, 0.0, ARMAS_LOWER, &conf);

  matrix_mult(&C0, &A, &At, alpha, 0.0, ARMAS_NULL, &conf);
  matrix_make_trm(&C0, ARMAS_LOWER);
  //ok = matrix_allclose(&C0, &C);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;

  printf("%6s: syrk(C, A, L|N) == TriL(gemm(C, A, A.T))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // 2. C = lower(C) + A.T*A
  matrix_set_values(&C, one, ARMAS_SYMM);
  matrix_mcopy(&C0, &C);

  matrix_make_trm(&C, ARMAS_LOWER);
  matrix_update_sym(&C, &At, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);

  matrix_mult(&C0, &At, &At, alpha, 0.0, ARMAS_TRANSA, &conf);
  matrix_make_trm(&C0, ARMAS_LOWER);
  //ok = matrix_allclose(&C0, &C);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;

  printf("%6s: syrk(C, A, L|T) == TriL(gemm(C, A.T, A))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  return fails;
}

// Local Variables:
// indent-tabs-mode: nil
// End:
