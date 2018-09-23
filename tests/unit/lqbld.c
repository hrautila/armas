
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqbld"

/*  -----------------------------------------------------------------------------------
 *  Test 5: unblk.Q(qr(A)) == blk.Q(qr(A))
 *    OK: ||unblk.Q(qr(A)) - blk.Q(qr(A))||_1 ~~ n*eps
 */
int test_build(int M, int N, int K, int lb, int verbose)
{
  char ct = N == K ? 'N' : 'K';
  armas_x_dense_t A0, A1, tau0;
  int ok;
  DTYPE n0;
  armas_conf_t conf = *armas_conf_default();
  
  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&tau0, imin(M, N), 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // factorize
  conf.lb = lb;
  armas_x_lqfactor(&A0, &tau0, &conf);
  armas_x_mcopy(&A1, &A0);
    
  // compute Q = buildQ(qr(A))
  conf.lb = 0;
  armas_x_lqbuild(&A0, &tau0, K, &conf);
  conf.lb = lb;
  armas_x_lqbuild(&A1, &tau0, K, &conf);

  n0 = rel_error((DTYPE *)0, &A0,   &A1,   ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(n0, N);
  printf("%s: unblk.Q(qr(A),%c) == blk.Q(qr(A),%c)\n", PASS(ok), ct, ct);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }

  armas_x_release(&A0);
  armas_x_release(&A1);
  armas_x_release(&tau0);
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
  armas_x_dense_t A0, C0, tau0, D;
  int ok;
  DTYPE n0;
  armas_conf_t conf = *armas_conf_default();
  armas_wbuf_t wb = ARMAS_WBNULL;

  armas_x_init(&A0, M, N);
  armas_x_init(&C0, M, M);
  armas_x_init(&tau0, imin(M, N), 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // factorize
  conf.lb = lb;
  armas_x_lqfactor(&A0, &tau0, &conf);

  // compute Q = buildQ(qr(A)), K first columns
  conf.lb = lb;
  armas_x_lqbuild(&A0, &tau0, K, &conf);

  // C0 = Q.T*Q - I
  armas_x_mult(0.0, &C0, 1.0, &A0, &A0, ARMAS_TRANSB, &conf);
  armas_x_diag(&D, &C0, 0);
  armas_x_add(&D, -1.0, &conf);

  n0 = armas_x_mnorm(&C0, ARMAS_NORM_ONE, &conf);

  ok = isOK(n0, N);
  printf("%s: %s Q(qr(A),%c).T * Q(qr(A),%c) == I\n", PASS(ok), blk, ct, ct);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }
  armas_x_release(&A0);
  armas_x_release(&C0);
  armas_x_release(&tau0);
  armas_wrelease(&wb);
  return ok;
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
