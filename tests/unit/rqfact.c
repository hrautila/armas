
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "lqfact"

/*  ----------------------------------------------------------------------------
 *  Test unblk.QR(A) == blk.QR(A)
 *     OK: ||unblk.QR(A) - blk.QR(A)||_1 < N*epsilon
 */
int test_factor(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, tau0, tau1, W, row;
  int wsize;
  double n0, n1;
  int wchange = lb > 8 ? 2*M : 0;
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
  wsize = armas_d_rqfactor_work(&A0, &conf);
  armas_d_init(&W, wsize-wchange, 1);

  // factorize
  conf.lb = 0;
  armas_d_rqfactor(&A0, &tau0, &W, &conf);

  conf.lb = lb;
  armas_d_rqfactor(&A1, &tau1, &W, &conf);

  n0 = rel_error((double *)0, &A1,   &A0,   ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  n1 = rel_error((double *)0, &tau1, &tau0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);

  printf("%s: unblk.RQ(A) == blk.RQ(A)\n", PASS(isOK(n0, N) && isOK(n1, N)));
  if (verbose > 0) {
    printf("  || error.RQ  ||_1: %e [%d]\n", n0, ndigits(n0));
    printf("  || error.tau ||_2: %e [%d]\n", n1, ndigits(n1));
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
  int N = 787;
  int M = 741;
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
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
