
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqsolve"

// test least squares min || B - A.T*X ||, M < N
//   1. compute B = A.T*X0
//   2. compute B =  A.-1*B
//   3. compute || X0 - B || == O(eps)
int test_lss(int M, int N, int K, int lb, int verbose)
{
  armas_x_dense_t A0, tau0;
  armas_x_dense_t B0, X0, X;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  DTYPE nrm;
  armas_wbuf_t wb = ARMAS_WBNULL;

  armas_x_init(&A0, M, N);
  armas_x_init(&B0, N, K);
  armas_x_init(&X0, M, K);
  armas_x_init(&tau0, M, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // set initial X
  armas_x_set_values(&X0, unitrand, ARMAS_ANY);

  // compute: B0 = A0.T*X0
  armas_x_mult(0.0, &B0, 1.0, &A0, &X0, ARMAS_TRANSA, &conf);

  conf.lb = lb;
  if (armas_x_lqsolve_w(&B0, &A0, &tau0, ARMAS_TRANS, &wb, &conf) != 0) {
    printf("solve: workspace calculation failure!!\n");
    return 0;
  }
  armas_walloc(&wb, wb.bytes);
#if 0
  wsize = armas_x_lqfactor_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);
#endif
  // factor
  armas_x_lqfactor_w(&A0, &tau0, &wb, &conf);

  // solve B0 = A.-T*B0
  armas_x_lqsolve_w(&B0, &A0, &tau0, ARMAS_TRANS, &wb, &conf);

  // X0 = X0 - A.-1*B0
  armas_x_submatrix(&X, &B0, 0, 0, M, K);

  nrm = rel_error((DTYPE *)0, &X, &X0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);
  printf("%s: min || B - A.T*X ||\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  armas_x_release(&A0);
  armas_x_release(&X0);
  armas_x_release(&B0);
  armas_x_release(&tau0);
  armas_wrelease(&wb);
  return ok;
}

// test: min || X || s.t. A*X = B
int test_min(int M, int N, int K, int lb, int verbose)
{
  armas_x_dense_t A0, A1, tau0;
  armas_x_dense_t B0, X0, B;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  DTYPE nrm, nrm0;
  armas_wbuf_t wb = ARMAS_WBNULL;
  
  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&B0, N, K);
  armas_x_init(&X0, N, K);
  armas_x_init(&tau0, N, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);

  // set B0
  armas_x_set_values(&B0, unitrand, ARMAS_ANY);
  nrm0 = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf);

  conf.lb = lb;
  if (armas_x_lqsolve_w(&B0, &A0, &tau0, ARMAS_TRANS, &wb, &conf) != 0) {
    printf("solve: workspace calculation failure!!\n");
    return 0;
  }
  armas_walloc(&wb, wb.bytes);
#if 0
  wsize = armas_x_qrfactor_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);
#endif
  // factor
  armas_x_lqfactor_w(&A0, &tau0, &wb, &conf);

  // X0 = A.-T*B0
  armas_x_mcopy(&X0, &B0);
  armas_x_lqsolve_w(&X0, &A0, &tau0, ARMAS_NONE, &wb, &conf);

  // B = B - A*X
  armas_x_submatrix(&B, &B0, 0, 0, M, K);
  armas_x_mult(1.0, &B, -1.0, &A1, &X0, ARMAS_NONE, &conf);

  nrm = armas_x_mnorm(&B, ARMAS_NORM_ONE, &conf) / nrm0;
  ok = isOK(nrm, N);
  printf("%s: min || X || s.t. A*X = B\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error || : %e [%d]\n", nrm, ndigits(nrm));
  }
  armas_x_release(&A0);
  armas_x_release(&A1);
  armas_x_release(&X0);
  armas_x_release(&B0);
  armas_x_release(&tau0);
  armas_wrelease(&wb);

  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int M = 787;
  int N = 741;
  int K = N;
  int LB = 48;
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
