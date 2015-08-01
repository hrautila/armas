
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "lqsolve"

// test least squares min || B - A.T*X ||, M < N
//   1. compute B = A.T*X0
//   2. compute B =  A.-1*B
//   3. compute || X0 - B || == O(eps)
int test_lss(int M, int N, int K, int lb, int verbose)
{
  armas_d_dense_t A0, A1, tau0;
  armas_d_dense_t B0, X0, W, X;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  double nrm, nrm0;

  armas_d_init(&A0, M, N);
  armas_d_init(&B0, N, K);
  armas_d_init(&X0, M, K);
  armas_d_init(&tau0, M, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // set initial X
  armas_d_set_values(&X0, unitrand, ARMAS_ANY);

  // compute: B0 = A0.T*X0
  armas_d_mult(&B0, &A0, &X0, 1.0, 0.0, ARMAS_TRANSA, &conf);

  conf.lb = lb;
  wsize = armas_d_lqfactor_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // factor
  armas_d_lqfactor(&A0, &tau0, &W, &conf);

  // solve B0 = A.-T*B0
  armas_d_lqsolve(&B0, &A0, &tau0, &W, ARMAS_TRANS, &conf);

  // X0 = X0 - A.-1*B0
  armas_d_submatrix(&X, &B0, 0, 0, M, K);

#if 0
  armas_d_scale_plus(&X0, &X, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&X0, ARMAS_NORM_ONE, &conf);
#endif
  nrm = rel_error((double *)0, &X, &X0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);
  printf("%s: min || B - A.T*X ||\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  return ok;
}

// test: min || X || s.t. A*X = B
int test_min(int M, int N, int K, int lb, int verbose)
{
  armas_d_dense_t A0, A1, tau0;
  armas_d_dense_t B0, X0, B1, W, B;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  double nrm, nrm0;

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&B0, N, K);
  armas_d_init(&X0, N, K);
  armas_d_init(&tau0, N, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  // set B0
  armas_d_set_values(&B0, unitrand, ARMAS_ANY);
  nrm0 = armas_d_mnorm(&B0, ARMAS_NORM_ONE, &conf);

  conf.lb = lb;
  wsize = armas_d_qrfactor_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // factor
  armas_d_lqfactor(&A0, &tau0, &W, &conf);

  // X0 = A.-T*B0
  armas_d_mcopy(&X0, &B0);
  armas_d_lqsolve(&X0, &A0, &tau0, &W, ARMAS_NONE, &conf);

  // B = B - A*X
  armas_d_submatrix(&B, &B0, 0, 0, M, K);
  armas_d_mult(&B, &A1, &X0, -1.0, 1.0, ARMAS_NONE, &conf);

  nrm = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf) / nrm0;
  ok = isOK(nrm, N);
  printf("%s: min || X || s.t. A*X = B\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error || : %e [%d]\n", nrm, ndigits(nrm));
  }
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

  // assert(M >= N)
  if (M < N) {
    int t = M;
    M = N;
    N = t;
  }

  int fails = 0;
  K = N/2;
  if (! test_lss(N, M, K, LB, verbose))
    fails++;
  if (! test_min(N, M, K, LB, verbose))
    fails++;
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
