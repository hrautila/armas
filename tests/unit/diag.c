
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "diag"

int test_left(int M, int N, int verbose)
{
  __Matrix A0, A1, D;
  int ok;
  __Dtype nrm;
  armas_conf_t conf = *armas_conf_default();

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&D, M, 1);

  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);
  matrix_set_values(&D, unitrand, ARMAS_ANY);

  matrix_mult_diag(&A1, &D, ARMAS_LEFT, &conf);
  if (verbose > 1 && N < 10) {
    printf("A0\n"); matrix_printf(stdout, "%6.3f", &A0);
    printf("A1\n"); matrix_printf(stdout, "%6.3f", &A1);
  }
  matrix_solve_diag(&A1, &D, ARMAS_LEFT, &conf);
#if 0
  matrix_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NOTRANS, &conf);
  if (verbose > 1 && N < 10) {
    printf("A1 - A0\n"); matrix_printf(stdout, "%6.3f", &A1);
  }
  nrm = matrix_mnorm(&A1, ARMAS_NORM_ONE, &conf);
#endif
  nrm = rel_error((__Dtype *)0, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok =isOK(nrm, N);
  printf("%4s: A = D.-1*D*A\n", PASS(ok));
  if (verbose > 0)
    printf("  M=%d, N=%d || rel error ||_1: %e\n", M, N, nrm);
  
  return ok;
}

int test_right(int M, int N, int verbose)
{
  __Matrix A0, A1, D;
  int ok;
  __Dtype nrm;
  armas_conf_t conf = *armas_conf_default();

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&D, N, 1);

  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);
  matrix_set_values(&D, unitrand, ARMAS_ANY);

  matrix_mult_diag(&A1, &D, ARMAS_RIGHT, &conf);
  if (verbose > 1 && N < 10) {
    printf("A0\n"); matrix_printf(stdout, "%6.3f", &A0);
    printf("A1\n"); matrix_printf(stdout, "%6.3f", &A1);
  }
  matrix_solve_diag(&A1, &D, ARMAS_RIGHT, &conf);
#if 0
  matrix_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NOTRANS, &conf);
  if (verbose > 1 && N < 10) {
    printf("A1 - A0\n"); matrix_printf(stdout, "%6.3f", &A1);
  }
  nrm = matrix_mnorm(&A1, ARMAS_NORM_ONE, &conf);
#endif
  nrm = rel_error((__Dtype *)0, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok =isOK(nrm, N);
  printf("%4s: A = A*D*D.-1\n", PASS(ok));
  if (verbose > 0)
    printf("  M=%d, N=%d || rel error ||_1: %e\n", M, N, nrm);
  
  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int M = 787;
  int N = 741;
  int verbose = 1;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose += 1;
      break;
    default:
      fprintf(stderr, "usage: %s [-v] [M N LB]\n", NAME);
      exit(1);
    }
  }
    
  if (optind < argc-1) {
    M = atoi(argv[optind]);
    N = atoi(argv[optind+1]);
  } else if (optind < argc) {
    N = atoi(argv[optind]);
    M = N; 
  }

  int fails = 0;
  if ( !test_left(M, N, verbose))
    fails++;
  if ( !test_right(M, N, verbose))
    fails++;
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
