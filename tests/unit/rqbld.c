
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqbld"

/*  -----------------------------------------------------------------------------------
 *  Test 5: unblk.Q(rq(A)) == blk.Q(rq(A))
 *    OK: ||unblk.Q(rq(A)) - blk.Q(rq(A))||_1 ~~ n*eps
 */
int test_build(int M, int N, int K, int lb, int verbose)
{
  char ct = N == K ? 'N' : 'K';
  __Matrix A0, A1, tau0, W;
  int wsize, ok;
  int wchange = lb > 0 ? 2*M : 0;
  __Dtype n0;
  armas_conf_t conf = *armas_conf_default();
  
  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&tau0, imin(M, N), 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = matrix_rqbuild_work(&A0, &conf);
  matrix_init(&W, wsize-wchange, 1);

  // factorize
  conf.lb = lb;
  matrix_rqfactor(&A0, &tau0, &W, &conf);
  matrix_mcopy(&A1, &A0);
    
  // compute Q = buildQ(rq(A))
  conf.lb = 0;
  matrix_rqbuild(&A0, &tau0, &W, K, &conf);
  conf.lb = lb;
  matrix_rqbuild(&A1, &tau0, &W, K, &conf);

  // ||A1 - A0||/||A0||
  n0 = rel_error((__Dtype *)0, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);

  ok = isOK(n0, N);
  printf("%s: unblk.Q(rq(A),%c) == blk.Q(rq(A),%c)\n", PASS(ok), ct, ct);
  if (verbose > 0) {
    printf("  || rel_error ||_1: %e [%d]\n", n0, ndigits(n0));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&W);
  matrix_release(&tau0);

  return ok;
}

/*  -----------------------------------------------------------------------------------
 *  Test 6: Q(qr(A)).T * Q(qr(A)) == I 
 *    OK: ||I - Q.T*Q||_1 ~~ n*eps
 */
int test_build_identity(int M, int N, int K, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  char ct = M == K ? 'M' : 'K';
  __Matrix A0, C0, tau0, D, W;
  int wsize, ok;
  __Dtype n0;
  armas_conf_t conf = *armas_conf_default();
  int wchange = lb > 0 ? 2*M : 0;

  matrix_init(&A0, M, N);
  matrix_init(&C0, M, M);
  matrix_init(&tau0, imin(M, N), 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = matrix_rqbuild_work(&A0, &conf);
  matrix_init(&W, wsize-wchange, 1);

  // factorize
  conf.lb = lb;
  matrix_rqfactor(&A0, &tau0, &W, &conf);

  // compute Q = buildQ(rq(A)), K first columns
  conf.lb = lb;
  matrix_rqbuild(&A0, &tau0, &W, K, &conf);

  // C0 = Q.T*Q - I
  matrix_mult(&C0, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  matrix_diag(&D, &C0, 0);
  matrix_add(&D, -1.0, &conf);

  n0 = matrix_mnorm(&C0, ARMAS_NORM_ONE, &conf);

  ok = isOK(n0, N);
  printf("%s: %s Q(rq(A),%c).T * Q(rq(A),%c) == I\n", PASS(ok), blk, ct, ct);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }
  //matrix_release(&A0);
  matrix_release(&C0);
  matrix_release(&tau0);

  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int N = 787;
  int M = 741;
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

  if (! test_build(M, N, M, LB, verbose))
    fails += 1;
  if (! test_build(M, N, M/2, LB, verbose))
    fails += 1;

  if (! test_build_identity(M, N, M, 0, verbose))
    fails += 1;
  if (! test_build_identity(M, N, M/2, 0, verbose))
    fails += 1;

  if (! test_build_identity(M, N, M, LB, verbose))
    fails += 1;
  if (! test_build_identity(M, N, M/2, LB, verbose))
    fails += 1;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
