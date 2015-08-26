
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
#define matrix_transpose  armas_s_transpose
#define matrix_release    armas_s_release
#define matrix_mult_trm   armas_s_mult_trm
#define matrix_mcopy      armas_s_mcopy

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define matrix_init       armas_d_init
#define matrix_set_values armas_d_set_values
#define matrix_mult       armas_d_mult
#define matrix_transpose  armas_d_transpose
#define matrix_release    armas_d_release
#define matrix_mult_trm   armas_d_mult_trm
#define matrix_mcopy      armas_d_mcopy

#endif
#include "helper.h"

int main(int argc, char **argv) {

  armas_conf_t conf;
  __Matrix C, B0, A, B;

  int ok, opt, fails = 0;
  int N = 600;
  int verbose = 1;
  __Dtype alpha = 1.0, n0, n1;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose++;
      break;
    default:
      fprintf(stderr, "usage: test_trmm [-v] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf = *armas_conf_default();

  matrix_init(&C, N, N);
  matrix_init(&A, N, N);
  matrix_init(&B, N, N);
  matrix_init(&B0, N, N);
  
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_set_values(&A, zero, ARMAS_NULL);
  matrix_set_values(&A, unitrand, ARMAS_UPPER);
  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_mcopy(&B0, &B);

  // B = A*B
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  matrix_mult(&C, &A, &B0, alpha, 0.0, ARMAS_NULL, &conf);

  n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmm(B, A, L|U)   == gemm(TriU(A), B)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_set_values(&B0, one, ARMAS_NULL);
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  matrix_mult(&C, &A, &B0, alpha, 0.0, ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmm(B, A, L|U|T) == gemm(TriU(A).T, B)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_set_values(&B0, one, ARMAS_NULL);
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  matrix_mult(&C, &B0, &A, alpha, 0.0, ARMAS_NULL, &conf);

  n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmm(B, A, R|U)   == gemm(B, TriU(A))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  matrix_set_values(&B, one, ARMAS_NULL);
  matrix_set_values(&B0, one, ARMAS_NULL);
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  matrix_mult(&C, &B0, &A, alpha, 0.0, ARMAS_TRANSB, &conf);

  n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmm(B, A, R|U|T) == gemm(B, TriU(A).T)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
