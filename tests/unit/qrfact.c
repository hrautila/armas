
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "qrfact"

/*  ----------------------------------------------------------------------------
 *  Test unblk.QR(A) == blk.QR(A)
 *     OK: ||unblk.QR(A) - blk.QR(A)||_1 < N*epsilon
 */
int test_factor(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1, tau0, tau1, W, row;
  int wsize;
  __Dtype n0, n1;
  int wchange = lb > 8 ? 2*M : 0;
  armas_conf_t conf = *armas_conf_default();
  
  if (lb == 0)
    lb = 4;

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&tau0, imin(M, N), 1);
  matrix_init(&tau1, imin(M, N), 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_NULL);
  matrix_mcopy(&A1, &A0);

  // allocate workspace according the blocked invocation
  conf.lb = lb;
  wsize = matrix_qrfactor_work(&A0, &conf);
  matrix_init(&W, wsize-wchange, 1);

  // factorize
  conf.lb = 0;
  matrix_qrfactor(&A0, &tau0, &W, &conf);

  conf.lb = lb;
  matrix_qrfactor(&A1, &tau1, &W, &conf);

  if (verbose > 1 && N < 10) {
    printf("unblk.QR(A):\n"); matrix_printf(stdout, "%9.2e", &A0);
    printf("unblk.tau:\n");   matrix_printf(stdout, "%9.2e", col_as_row(&row, &tau0));
    printf("  blk.tau:\n");   matrix_printf(stdout, "%9.2e", col_as_row(&row, &tau1));
    printf("  blk.QR(A):\n"); matrix_printf(stdout, "%9.2e", &A1);
  }

  n0 = rel_error((__Dtype *)0, &A0,   &A1,   ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  n1 = rel_error((__Dtype *)0, &tau0, &tau1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);

  printf("%s: unblk.QR(A) == blk.QR(A)\n", PASS(isOK(n0, N) && isOK(n1, N)));
  if (verbose > 0) {
    printf("  || error.QR  ||_1: %e [%d]\n", n0, ndigits(n0));
    printf("  || error.tau ||_2: %e [%d]\n", n1, ndigits(n1));
  }
  
  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&tau0);
  matrix_release(&tau1);

  return isOK(n0, N) && isOK(n1, N);
}

int main(int argc, char **argv)
{
  int opt;
  int M = 787;
  int N = 741;
  int LB = 36;
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
    fails += 1;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
