
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
  armas_x_dense_t A0, A1, D;
  int ok;
  DTYPE nrm;
  armas_conf_t conf = *armas_conf_default();

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&D, M, 1);

  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);
  armas_x_set_values(&D, unitrand, ARMAS_ANY);

  armas_x_mult_diag(&A1, &D, ARMAS_LEFT, &conf);
  if (verbose > 1 && N < 10) {
    printf("A0\n"); armas_x_printf(stdout, "%6.3f", &A0);
    printf("A1\n"); armas_x_printf(stdout, "%6.3f", &A1);
  }
  armas_x_solve_diag(&A1, &D, ARMAS_LEFT, &conf);
#if 0
  armas_x_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NOTRANS, &conf);
  if (verbose > 1 && N < 10) {
    printf("A1 - A0\n"); armas_x_printf(stdout, "%6.3f", &A1);
  }
  nrm = armas_x_mnorm(&A1, ARMAS_NORM_ONE, &conf);
#endif
  nrm = rel_error((DTYPE *)0, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok =isOK(nrm, N);
  printf("%4s: A = D.-1*D*A\n", PASS(ok));
  if (verbose > 0)
    printf("  M=%d, N=%d || rel error ||_1: %e\n", M, N, nrm);
  
  return ok;
}

int test_right(int M, int N, int verbose)
{
  armas_x_dense_t A0, A1, D;
  int ok;
  DTYPE nrm;
  armas_conf_t conf = *armas_conf_default();

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&D, N, 1);

  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);
  armas_x_set_values(&D, unitrand, ARMAS_ANY);

  armas_x_mult_diag(&A1, &D, ARMAS_RIGHT, &conf);
  if (verbose > 1 && N < 10) {
    printf("A0\n"); armas_x_printf(stdout, "%6.3f", &A0);
    printf("A1\n"); armas_x_printf(stdout, "%6.3f", &A1);
  }
  armas_x_solve_diag(&A1, &D, ARMAS_RIGHT, &conf);
#if 0
  armas_x_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NOTRANS, &conf);
  if (verbose > 1 && N < 10) {
    printf("A1 - A0\n"); armas_x_printf(stdout, "%6.3f", &A1);
  }
  nrm = armas_x_mnorm(&A1, ARMAS_NORM_ONE, &conf);
#endif
  nrm = rel_error((DTYPE *)0, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
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
