
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "qrbld"


/*  -----------------------------------------------------------------------------------
 *  Test 5: unblk.Q(qr(A)) == blk.Q(qr(A))
 *    OK: ||unblk.Q(qr(A)) - blk.Q(qr(A))||_1 ~~ n*eps
 */
int test_qrbuild(int M, int N, int K, int lb, int verbose)
{
  char ct = N == K ? 'N' : 'K';
  __Matrix A0, A1, tau0, W;
  int wsize, ok;
  __Dtype n0, n1;
  int wchange = lb > 8 ? 2*M : 0;
  armas_conf_t conf = *armas_conf_default();
  
  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&tau0, imin(M, N), 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = matrix_qrbuild_work(&A0, &conf);
  matrix_init(&W, wsize-wchange, 1);

  // factorize
  conf.lb = lb;
  matrix_qrfactor(&A0, &tau0, &W, &conf);
  matrix_mcopy(&A1, &A0);
  if (verbose > 1) {
    printf("qr(A):\n"); matrix_printf(stdout, "%9.2e", &A1);
  }
    
  // compute Q = buildQ(qr(A))
  conf.lb = 0;
  matrix_qrbuild(&A0, &tau0, &W, K, &conf);
  conf.lb = lb;
  matrix_qrbuild(&A1, &tau0, &W, K, &conf);
  if (verbose > 1) {
    printf("unblk.Q(qr(A)):\n"); matrix_printf(stdout, "%9.2e", &A0);
    printf("  blk.Q(qr(A)):\n"); matrix_printf(stdout, "%9.2e", &A1);
  }

  n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(n0, N);
  printf("%s: unblk.Q(qr(A),%c) == blk.Q(qr(A),%c)\n", PASS(ok), ct, ct);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&tau0);

  return ok;
}

/*  -----------------------------------------------------------------------------------
 *  Test 6: Q(qr(A)).T * Q(qr(A)) == I 
 *    OK: ||I - Q.T*Q||_1 ~~ n*eps
 */
int test_qrbuild_identity(int M, int N, int K, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  char ct = N == K ? 'N' : 'K';
  __Matrix A0, C0, C1, tau0, D, W;
  int wsize, ok;
  __Dtype n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  matrix_init(&A0, M, N);
  matrix_init(&C0, N, N);
  matrix_init(&C1, N, N);
  matrix_init(&tau0, imin(M, N), 1);
  matrix_set_values(&C1, zero, ARMAS_ANY);
  matrix_diag(&D, &C1, 0);
  matrix_add(&D, 1.0, &conf);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = matrix_qrbuild_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  matrix_qrfactor(&A0, &tau0, &W, &conf);

  // compute Q = buildQ(qr(A)), K first columns
  conf.lb = lb;
  matrix_qrbuild(&A0, &tau0, &W, K, &conf);

  // C0 = Q.T*Q 
  matrix_mult(&C0, &A0, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(n0, N);
  printf("%s: %s Q(qr(A),%c).T * Q(qr(A),%c) == I\n", PASS(ok), blk, ct, ct);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }
  matrix_release(&A0);
  matrix_release(&C0);
  matrix_release(&C1);
  matrix_release(&tau0);

  return ok;
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
  if (! test_qrbuild(M, N, N, LB, verbose))
    fails += 1;
  if (! test_qrbuild(M, N, N/2, LB, verbose))
    fails += 1;

  if (! test_qrbuild_identity(M, N, N, 0, verbose))
    fails += 1;
  if (! test_qrbuild_identity(M, N, N/2, 0, verbose))
    fails += 1;

  if (! test_qrbuild_identity(M, N, N, LB, verbose))
    fails += 1;
  if (! test_qrbuild_identity(M, N, N/2, LB, verbose))
    fails += 1;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
