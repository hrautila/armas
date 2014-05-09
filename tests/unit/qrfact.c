
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "qrfact"

/*  ----------------------------------------------------------------------------
 *  Test unblk.QR(A) == blk.QR(A)
 *     OK: ||unblk.QR(A) - blk.QR(A)||_1 < N*epsilon
 */
int test_factor(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, tau0, tau1, W, row;
  int wsize;
  double n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  if (lb == 0)
    lb = 4;

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&tau0, imin(M, N), 1);
  armas_d_init(&tau1, imin(M, N), 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_NULL);
  armas_d_mcopy(&A1, &A0);

  // allocate workspace according the blocked invocation
  conf.lb = lb;
  wsize = armas_d_qrfactor_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // factorize
  conf.lb = 0;
  armas_d_qrfactor(&A0, &tau0, &W, &conf);

  conf.lb = lb;
  armas_d_qrfactor(&A1, &tau1, &W, &conf);

  if (verbose > 1 && N < 10) {
    printf("unblk.QR(A):\n"); armas_d_printf(stdout, "%9.2e", &A0);
    printf("unblk.tau:\n");   armas_d_printf(stdout, "%9.2e", col_as_row(&row, &tau0));
    printf("  blk.tau:\n");   armas_d_printf(stdout, "%9.2e", col_as_row(&row, &tau1));
    printf("  blk.QR(A):\n"); armas_d_printf(stdout, "%9.2e", &A1);
  }

  // A0 = A0 - A1
  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  n0 = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
  // tau0 = tau0 - tau1
  armas_d_axpy(&tau0, &tau1, -1.0, &conf);
  n1 = armas_d_nrm2(&tau0, &conf);

  printf("%s: unblk.QR(A) == blk.QR(A)\n", PASS(isOK(n0, N) && isOK(n1, N)));
  if (verbose > 0) {
    printf("  || error.QR  ||_1: %e [%ld]\n", n0, (int64_t)(n0/DBL_EPSILON));
    printf("  || error.tau ||_2: %e [%ld]\n", n1, (int64_t)(n1/DBL_EPSILON));
  }
  
  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&tau0);
  armas_d_release(&tau1);

  return isOK(n0, N) && isOK(n1, N);
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
  if (! test_factor(M, N, LB, verbose))
    fails += 1;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End: