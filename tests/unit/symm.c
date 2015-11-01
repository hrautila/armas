
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int main(int argc, char **argv) {

  int ok, opt, verbose = 1, fails = 0;
  int N = 633;
  armas_conf_t conf;
  __Matrix C0, C1, A, As,  B;
  __Dtype n0, n1;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose++;
      break;
    default:
      fprintf(stderr, "usage: symm [-P nproc] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf = *armas_conf_default();

  matrix_init(&C0, N, N);
  matrix_init(&C1, N, N);
  matrix_init(&A, N, N);
  matrix_init(&As, N, N);
  matrix_init(&B, N, N);
  
  matrix_set_values(&C0, zero, ARMAS_NULL);
  matrix_set_values(&C1, zero, ARMAS_NULL);
  matrix_set_values(&A, unitrand, ARMAS_SYMM);
  matrix_set_values(&B, unitrand, ARMAS_NULL);
  
#if 0
  if (verbose > 2 && N < 10) {
    printf("A\n"); matrix_printf(stdout, "%4.2f", &A);
    printf("B\n"); matrix_printf(stdout, "%4.2f", &B);
  }
#endif
  // C0 = symm(upper(A)*B);  C1 = A*B
  matrix_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_LEFT|ARMAS_UPPER, &conf);
  matrix_mult(&C1, &A, &B, 1.0, 0.0, 0, &conf);

  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

  printf("%6s: symm(upper(A), B) == gemm(A, B)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // C = B*upper(A)
  matrix_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_RIGHT|ARMAS_UPPER, &conf);
  matrix_mult(&C1, &B, &A, 1.0, 0.0, 0, &conf);

  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

  printf("%6s: symm(B, upper(A)) == gemm(B, A)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // LOWER
  matrix_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_LEFT|ARMAS_LOWER, &conf);
  matrix_mult(&C1, &A, &B, 1.0, 0.0, 0, &conf);
  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

  printf("%6s: symm(lower(A), B) == gemm(A, B)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;
  if (! ok)
    exit(1 - ok);

  // C = B*upper(A)
  matrix_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_RIGHT|ARMAS_LOWER, &conf);
  matrix_mult(&C1, &B, &A, 1.0, 0.0, 0, &conf);
  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

  printf("%6s: symm(B, lower(A)) == gemm(B, A)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
