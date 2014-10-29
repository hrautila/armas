
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "diag"

int test_left(int M, int N, int verbose)
{
  armas_d_dense_t A0, A1, D;
  int ok;
  double nrm;
  armas_conf_t conf = *armas_conf_default();

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&D, M, 1);

  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);
  armas_d_set_values(&D, unitrand, ARMAS_ANY);

  armas_d_mult_diag(&A1, &D, ARMAS_LEFT, &conf);
  if (verbose > 1 && N < 10) {
    printf("A0\n"); armas_d_printf(stdout, "%6.3f", &A0);
    printf("A1\n"); armas_d_printf(stdout, "%6.3f", &A1);
  }
  armas_d_solve_diag(&A1, &D, ARMAS_LEFT, &conf);
  armas_d_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NOTRANS, &conf);
  if (verbose > 1 && N < 10) {
    printf("A1 - A0\n"); armas_d_printf(stdout, "%6.3f", &A1);
  }
  nrm = armas_d_mnorm(&A1, ARMAS_NORM_ONE, &conf);
  ok =isOK(nrm, N);
  printf("%4s: A = D.-1*D*A\n", PASS(ok));
  if (verbose > 0)
    printf("  M=%d, N=%d ||A - D.-1*D*A||_1: %e\n", M, N, nrm);
  
  return ok;
}

int test_right(int M, int N, int verbose)
{
  armas_d_dense_t A0, A1, D;
  int ok;
  double nrm;
  armas_conf_t conf = *armas_conf_default();

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&D, N, 1);

  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);
  armas_d_set_values(&D, unitrand, ARMAS_ANY);

  armas_d_mult_diag(&A1, &D, ARMAS_RIGHT, &conf);
  if (verbose > 1 && N < 10) {
    printf("A0\n"); armas_d_printf(stdout, "%6.3f", &A0);
    printf("A1\n"); armas_d_printf(stdout, "%6.3f", &A1);
  }
  armas_d_solve_diag(&A1, &D, ARMAS_RIGHT, &conf);
  armas_d_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NOTRANS, &conf);
  if (verbose > 1 && N < 10) {
    printf("A1 - A0\n"); armas_d_printf(stdout, "%6.3f", &A1);
  }
  nrm = armas_d_mnorm(&A1, ARMAS_NORM_ONE, &conf);
  ok =isOK(nrm, N);
  printf("%4s: A = A*D*D.-1\n", PASS(ok));
  if (verbose > 0)
    printf("  M=%d, N=%d ||A - A*D*D.-1||_1: %e\n", M, N, nrm);
  
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 787;
  int N = 741;
  int K = N;
  int LB = 36;
  int ok = 0;
  int nproc = 1;
  int verbose = 1;

  while ((opt = getopt(argc, argv, "P:v")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    case 'v':
      verbose += 1;
      break;
    default:
      fprintf(stderr, "usage: %s [-v] [M N LB]\n", NAME);
      exit(1);
    }
  }
    
  if (optind < argc-2) {
    M = atoi(argv[optind]);
    N = atoi(argv[optind+1]);
    LB = atoi(argv[optind+2]);
  } else if (optind < argc-1) {
    N = atoi(argv[optind]);
    M = N;
    LB = atoi(argv[optind+1]);
  } else if (optind < argc) {
    N = atoi(argv[optind]);
    M = N; LB = 0;
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
