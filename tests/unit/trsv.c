
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_row_vector(int N, int verbose, int flags)
{
  armas_conf_t conf = *armas_conf_default();
  armas_d_dense_t Z, X, A, X0;
  int ok, fails = 0;
  double n0, n1;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  
  armas_d_init(&Z, N+2, N);
  armas_d_row(&X0, &Z, 0);
  armas_d_row(&X,  &Z, 1);
  armas_d_submatrix(&A, &Z, 2, 0, N, N);

  armas_d_set_values(&X0, one, ARMAS_NULL);
  armas_d_set_values(&X, one, ARMAS_NULL);
  armas_d_set_values(&A, zeromean, flags);

  if (verbose > 1 && N < 10) {
    printf("Z:\n"); armas_d_printf(stdout, "%6.3f", &Z);
  }

  armas_d_mvmult(0.0, &X, 1.0, &A, &X0, 0, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, flags, &conf);
  n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  if (verbose > 2 && N < 10) {
    printf("X-X0:\n"); armas_d_printf(stdout, "%6.3f", &X);
  }
  printf("%6s : row(X) = trsv(trmv(X, A, %c|N), A, %c|N)\n", PASS(ok), uplo, uplo);
  fails += 1 - ok;
  if (verbose > 0) {
    printf( "  || error ||: %e\n", n0);
  }

  armas_d_mvmult(0.0, &X, 1.0, &A, &X0, ARMAS_TRANS, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, flags|ARMAS_TRANS, &conf);
  n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  if (verbose > 2 && N < 10) {
    printf("X-X0:\n"); armas_d_printf(stdout, "%6.3f", &X);
  }
  printf("%6s : row(X) = trsv(trmv(X, A, %c|T), A, %c|T)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf( "  || error ||: %e\n", n0);
  }
  fails += 1 - ok;

  return fails;
}


int test_col_vector(int N, int verbose, int flags)
{
  armas_conf_t conf = *armas_conf_default();
  armas_d_dense_t Z, X, A, X0, tmp;
  int ok, fails = 0;
  double n0, n1;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  
  armas_d_init(&Z, N, N+2);
  armas_d_column(&X0, &Z, 0);
  armas_d_column(&X,  &Z, 1);
  armas_d_submatrix(&A, &Z, 0, 2, N, N);

  armas_d_set_values(&X0, one, ARMAS_NULL);
  armas_d_set_values(&X, one, ARMAS_NULL);
  armas_d_set_values(&A, zeromean, flags);

  if (verbose > 1 && N < 10) {
    printf("Z:\n"); armas_d_printf(stdout, "%6.3f", &Z);
  }

  armas_d_mvmult(0.0, &X, 1.0, &A, &X0, 0, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, flags, &conf);
  n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  if (verbose > 2 && N < 10) {
    printf("X-X0:\n"); armas_d_printf(stdout, "%6.3f", armas_d_col_as_row(&tmp, &X));
  }
  printf("%6s : col(X) = trsv(trmv(X, A, %c|N), A, %c|N)\n", PASS(ok), uplo, uplo);
  fails += 1 - ok;
  if (verbose > 0) {
    printf( "  || error ||: %e\n", n0);
  }

  armas_d_mvmult(0.0, &X, 1.0, &A, &X0, ARMAS_TRANS, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, flags|ARMAS_TRANS, &conf);
  n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  if (verbose > 2 && N < 10) {
    printf("X-X0:\n"); armas_d_printf(stdout, "%6.3f", armas_d_col_as_row(&tmp, &X));
  }
  printf("%6s : col(X) = trsv(trmv(X, A, %c|T), A, %c|T)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf( "  || error ||: %e\n", n0);
  }
  fails += 1 - ok;

  return fails;
}

int main(int argc, char **argv) 
{
  int opt;
  int N = 77;
  int verbose = 0;
  
  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose ++;
      break;
    default:
      fprintf(stderr, "usage: trsv [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc) {
     N = atoi(argv[optind]);
  }

  int fails = 0;

  fails += test_row_vector(N, verbose, ARMAS_UPPER);
  fails += test_col_vector(N, verbose, ARMAS_UPPER);
  fails += test_row_vector(N, verbose, ARMAS_LOWER);
  fails += test_col_vector(N, verbose, ARMAS_LOWER);
  
  
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
