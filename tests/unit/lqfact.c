
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqfact"

/*  ----------------------------------------------------------------------------
 *  Test unblk.QR(A) == blk.QR(A)
 *     OK: ||unblk.QR(A) - blk.QR(A)||_1 < N*epsilon
 */
int test_factor(int M, int N, int lb, int verbose)
{
  armas_x_dense_t A0, A1, tau0, tau1;
  DTYPE n0, n1;
  armas_conf_t conf = *armas_conf_default();
  armas_wbuf_t wb = ARMAS_WBNULL;
  
  if (lb == 0)
    lb = 4;

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&tau0, imin(M, N), 1);
  armas_x_init(&tau1, imin(M, N), 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_NULL);
  armas_x_mcopy(&A1, &A0);

  // allocate workspace according the blocked invocation
  conf.lb = lb;
  if (armas_x_lqfactor_w(&A0, &tau0, &wb, &conf) != 0) {
    printf("factor: workspace calculation failure!!\n");
    return 0;
  }
  armas_walloc(&wb, wb.bytes);

  // factorize
  conf.lb = 0;
  armas_x_lqfactor_w(&A0, &tau0, &wb, &conf);

  conf.lb = lb;
  armas_x_lqfactor_w(&A1, &tau1, &wb, &conf);

  n0 = rel_error((DTYPE *)0, &A0,   &A1,   ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  n1 = rel_error((DTYPE *)0, &tau0, &tau1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);

  printf("%s: unblk.LQ(A) == blk.LQ(A)\n", PASS(isOK(n0, N) && isOK(n1, N)));
  if (verbose > 0) {
    printf("  || error.LQ  ||_1: %e [%d]\n", n0, ndigits(n0));
    printf("  || error.tau ||_2: %e [%d]\n", n1, ndigits(n1));
  }
  
  armas_x_release(&A0);
  armas_x_release(&A1);
  armas_x_release(&tau0);
  armas_x_release(&tau1);
  armas_wrelease(&wb);
  return isOK(n0, N) && isOK(n1, N);
}


int main(int argc, char **argv)
{
  int opt;
  int N = 787;
  int M = 741;
  int LB = 64;
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

  int fails = 0;
  if (! test_factor(M, N, LB, verbose))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
