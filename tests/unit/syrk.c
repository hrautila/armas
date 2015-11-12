
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

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
  if (verbose > 1 && N < 10) {
    printf("syrk(C)\n"); matrix_printf(stdout, "%5.2f", &C);
    printf("C0\n"); matrix_printf(stdout, "%5.2f", &C0);
  }

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

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

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

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

  if (verbose > 1 && N < 10) {
    printf("syrk(C)\n"); matrix_printf(stdout, "%5.2f", &C);
    printf("C0\n"); matrix_printf(stdout, "%5.2f", &C0);
  }
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

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

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

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
