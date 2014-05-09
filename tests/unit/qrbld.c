
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "qrbld"


/*  -----------------------------------------------------------------------------------
 *  Test 5: unblk.Q(qr(A)) == blk.Q(qr(A))
 *    OK: ||unblk.Q(qr(A)) - blk.Q(qr(A))||_1 ~~ n*eps
 */
int test_qrbuild(int M, int N, int K, int lb, int verbose)
{
  char ct = N == K ? 'N' : 'K';
  armas_d_dense_t A0, A1, C0, tau0, W;
  int wsize;
  double n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&tau0, imin(M, N), 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_d_qrbuild_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  armas_d_qrfactor(&A0, &tau0, &W, &conf);
  armas_d_mcopy(&A1, &A0);
  if (verbose > 1) {
    printf("qr(A):\n"); armas_d_printf(stdout, "%9.2e", &A1);
  }
    
  // compute Q = buildQ(qr(A))
  conf.lb = 0;
  armas_d_qrbuild(&A0, &tau0, &W, K, &conf);
  conf.lb = lb;
  armas_d_qrbuild(&A1, &tau0, &W, K, &conf);
  if (verbose > 1) {
    printf("unblk.Q(qr(A)):\n"); armas_d_printf(stdout, "%9.2e", &A0);
    printf("  blk.Q(qr(A)):\n"); armas_d_printf(stdout, "%9.2e", &A1);
  }

  // A1 = A1 - A0
  armas_d_scale_plus(&A1, &A0, 1.0, -1.0, ARMAS_NONE, &conf);
  n0 = armas_d_mnorm(&A1, ARMAS_NORM_ONE, &conf);

  printf("%s: unblk.Q(qr(A),%c) == blk.Q(qr(A),%c)\n", PASS(isOK(n0, N)), ct, ct);
  if (verbose > 0) {
    printf("  || error ||_1: %e [%ld]\n", n0, (int64_t)(n0/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&tau0);

  return isOK(n0, N);
}

/*  -----------------------------------------------------------------------------------
 *  Test 6: Q(qr(A)).T * Q(qr(A)) == I 
 *    OK: ||I - Q.T*Q||_1 ~~ n*eps
 */
int test_qrbuild_identity(int M, int N, int K, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  char ct = N == K ? 'N' : 'K';
  armas_d_dense_t A0, C0, tau0, D, W;
  int wsize;
  double n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  armas_d_init(&A0, M, N);
  armas_d_init(&C0, N, N);
  armas_d_init(&tau0, imin(M, N), 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_d_qrbuild_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  armas_d_qrfactor(&A0, &tau0, &W, &conf);

  // compute Q = buildQ(qr(A)), K first columns
  conf.lb = lb;
  armas_d_qrbuild(&A0, &tau0, &W, K, &conf);

  // C0 = Q.T*Q - I
  armas_d_mult(&C0, &A0, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
  armas_d_diag(&D, &C0, 0);
  armas_d_add(&D, -1.0, &conf);

  n0 = armas_d_mnorm(&C0, ARMAS_NORM_ONE, &conf);

  printf("%s: %s Q(qr(A),%c).T * Q(qr(A),%c) == I\n", PASS(isOK(n0, N)), blk, ct, ct);
  if (verbose > 0) {
    printf("  || error ||_1: %e [%ld]\n", n0, (int64_t)(n0/DBL_EPSILON));
  }
  armas_d_release(&A0);
  armas_d_release(&C0);
  armas_d_release(&tau0);

  return isOK(n0, N);
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
