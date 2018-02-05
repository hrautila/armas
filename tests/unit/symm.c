
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int main(int argc, char **argv) {

  int ok, opt, verbose = 1, fails = 0;
  int N = 633;
  armas_conf_t conf;
  armas_x_dense_t C0, C1, A, As,  B;
  DTYPE n0, n1;

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

  armas_x_init(&C0, N, N);
  armas_x_init(&C1, N, N);
  armas_x_init(&A, N, N);
  armas_x_init(&As, N, N);
  armas_x_init(&B, N, N);
  
  armas_x_set_values(&C0, zero, ARMAS_NULL);
  armas_x_set_values(&C1, zero, ARMAS_NULL);
  armas_x_set_values(&A, unitrand, ARMAS_SYMM);
  armas_x_set_values(&B, unitrand, ARMAS_NULL);
  
#if 0
  if (verbose > 2 && N < 10) {
    printf("A\n"); armas_x_printf(stdout, "%4.2f", &A);
    printf("B\n"); armas_x_printf(stdout, "%4.2f", &B);
  }
#endif
  // C0 = symm(upper(A)*B);  C1 = A*B
  armas_x_mult_sym(0.0, &C0, 1.0, &A, &B, ARMAS_LEFT|ARMAS_UPPER, &conf);
  armas_x_mult(0.0, &C1, 1.0, &A, &B, 0, &conf);

  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

  printf("%6s: symm(upper(A), B) == gemm(A, B)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // C = B*upper(A)
  armas_x_mult_sym(0.0, &C0, 1.0, &A, &B, ARMAS_RIGHT|ARMAS_UPPER, &conf);
  armas_x_mult(0.0, &C1, 1.0, &B, &A, 0, &conf);

  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

  printf("%6s: symm(B, upper(A)) == gemm(B, A)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // LOWER
  armas_x_mult_sym(0.0, &C0, 1.0, &A, &B, ARMAS_LEFT|ARMAS_LOWER, &conf);
  armas_x_mult(0.0, &C1, 1.0, &A, &B, 0, &conf);
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
  armas_x_mult_sym(0.0, &C0, 1.0, &A, &B, ARMAS_RIGHT|ARMAS_LOWER, &conf);
  armas_x_mult(0.0, &C1, 1.0, &B, &A, 0, &conf);
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
