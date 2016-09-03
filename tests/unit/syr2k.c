
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int main(int argc, char **argv) 
{

  armas_conf_t conf;
  armas_x_dense_t C, C0, A, At, B, Bt;

  int ok, opt;
  int N = 8;
  int verbose = 1;
  int fails = 0;
  DTYPE n0, n1, alpha = 1.0;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose++;
      break;
    default:
      fprintf(stderr, "usage: syr2k [-v] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf = *armas_conf_default();

  armas_x_init(&C,  N, N);
  armas_x_init(&C0, N, N);
  armas_x_init(&A,  N, N/2);
  armas_x_init(&At, N/2, N);
  armas_x_init(&B,  N, N/2);
  armas_x_init(&Bt, N/2, N);
  
  armas_x_set_values(&A, zeromean, ARMAS_NULL);
  armas_x_set_values(&B, zeromean, ARMAS_NULL);
  armas_x_transpose(&At, &A);
  armas_x_transpose(&Bt, &B);

  // 1. C = upper(C) + A*B.T + B*A.T;
  armas_x_set_values(&C, one, ARMAS_SYMM);
  armas_x_mcopy(&C0, &C);

  armas_x_make_trm(&C, ARMAS_UPPER);
  armas_x_update2_sym(&C, &A, &B, alpha, 0.0, ARMAS_UPPER, &conf);

  armas_x_mult(&C0, &A, &Bt, alpha, 0.0, ARMAS_NULL, &conf);
  armas_x_mult(&C0, &B, &At, alpha, 1.0, ARMAS_NULL, &conf);
  armas_x_make_trm(&C0, ARMAS_UPPER);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NULL, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: syr2k(C, A, U|N) == TriU(C + A*B.T + B*A.T))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;
  
  // 2. C = upper(C) + B.T*A + A,T*B
  armas_x_set_values(&C, one, ARMAS_SYMM);
  armas_x_mcopy(&C0, &C);

  armas_x_make_trm(&C, ARMAS_UPPER);
  armas_x_update2_sym(&C, &At, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);

  armas_x_mult(&C0, &Bt, &A, alpha, 0.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  armas_x_mult(&C0, &At, &B, alpha, 1.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  armas_x_make_trm(&C0, ARMAS_UPPER);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NULL, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: syr2k(C, A, T|N) == TriU(C + B.T*A + A.T*B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // 1. C = lower(C) + A*A.T;
  armas_x_set_values(&C, one, ARMAS_SYMM);
  armas_x_mcopy(&C0, &C);

  armas_x_make_trm(&C, ARMAS_LOWER);
  armas_x_update2_sym(&C, &A, &B, alpha, 0.0, ARMAS_LOWER, &conf);

  armas_x_mult(&C0, &A, &Bt, alpha, 0.0, ARMAS_NULL, &conf);
  armas_x_mult(&C0, &B, &At, alpha, 1.0, ARMAS_NULL, &conf);
  armas_x_make_trm(&C0, ARMAS_LOWER);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NULL, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: syr2k(C, A, L|N) == TriL(C + A*B.T + B*A.T))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // 2. C = lower(C) + A.T*A
  armas_x_set_values(&C, one, ARMAS_SYMM);
  armas_x_mcopy(&C0, &C);

  armas_x_make_trm(&C, ARMAS_LOWER);
  armas_x_update2_sym(&C, &At, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);

  armas_x_mult(&C0, &Bt, &A, alpha, 0.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  armas_x_mult(&C0, &At, &B, alpha, 1.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  armas_x_make_trm(&C0, ARMAS_LOWER);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NULL, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;

  printf("%6s: syr2k(C, A, T|N) == TriL(gemm(C + B.T*A + A.T*B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
