
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqmult"

/*  ------------------------------------------------------------------------
 *  Test 2: I - Q.T*Q = 0.0
 *    OK: ||I - Q.T*Q||_1 ~~ n*eps
 *  
 */
int test_mult_identity(int M, int N, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  armas_x_dense_t A0, C, tau0, W, D;
  int wsize;
  DTYPE n0;
  int wchange = lb > 8 ? 2*M : 0;
  armas_conf_t conf = *armas_conf_default();
  
  armas_x_init(&A0, M, N);
  armas_x_init(&C, N, M);
  armas_x_init(&tau0, imin(M, N), 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // C is first M columns of identity matrix
  armas_x_set_values(&C, zero, ARMAS_ANY);
  armas_x_diag(&D, &C, 0);
  armas_x_add(&D, 1.0, &conf);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_x_lqmult_work(&C, ARMAS_LEFT, &conf);
  armas_x_init(&W, wsize-wchange, 1);

  // factorize
  conf.lb = lb;
  armas_x_lqfactor(&A0, &tau0, &W, &conf);

  conf.error = 0;
  armas_x_lqmult(&C, &A0, &tau0, &W, ARMAS_LEFT, &conf);
  armas_x_lqmult(&C, &A0, &tau0, &W, ARMAS_LEFT|ARMAS_TRANS, &conf);

  // subtract 1.0 on diagonal
  armas_x_add(&D, -1.0, &conf);
  n0 = armas_x_mnorm(&C, ARMAS_NORM_ONE, &conf);

  printf("%s: %s Q.T*Q == I\n", PASS(isOK(n0, N)), blk);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }
  armas_x_release(&A0);
  armas_x_release(&C);
  armas_x_release(&tau0);
  return isOK(n0, N);
}

/*  ----------------------------------------------------------------------------
 *  Test 3: C == Q.T*Q*C
 *    OK: ||C - Q.T*Q*C||_1 ~~ n*eps
 */
int test_mult_left(int M, int N, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  armas_x_dense_t A0, C1, C0, tau0, W;
  int wsize;
  DTYPE n0;
  armas_conf_t conf = *armas_conf_default();
  
  armas_x_init(&A0, M, N);
  armas_x_init(&C0, N, M);
  armas_x_init(&C1, N, M);
  armas_x_init(&tau0, imin(M, N), 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // set C 
  armas_x_set_values(&C0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&C1, &C0);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_x_lqmult_work(&C0, ARMAS_LEFT, &conf);
  armas_x_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  armas_x_lqfactor(&A0, &tau0, &W, &conf);

  conf.error = 0;
  // compute C0 = Q.T*Q*C0
  armas_x_lqmult(&C0, &A0, &tau0, &W, ARMAS_LEFT, &conf);
  armas_x_lqmult(&C0, &A0, &tau0, &W, ARMAS_LEFT|ARMAS_TRANS, &conf);

  n0 = rel_error((DTYPE *)0, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);

  printf("%s: %s Q.T*Q*C == C\n", PASS(isOK(n0, N)), blk);
  if (verbose > 0) {
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
  }

  armas_x_release(&A0);
  armas_x_release(&C0);
  armas_x_release(&C1);
  armas_x_release(&W);
  armas_x_release(&tau0);
  return isOK(n0, N);
}


/*  -----------------------------------------------------------------------------------
 *  Test 4: C == C*Q.T*Q
 *    OK: ||C - C*Q.T*Q||_1 ~~ n*eps
 */
int test_mult_right(int M, int N, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  armas_x_dense_t A0, C1, C0, tau0, W;
  int wsize;
  DTYPE n0;
  armas_conf_t conf = *armas_conf_default();
  
  armas_x_init(&A0, M, N);
  armas_x_init(&C0, M, N);
  armas_x_init(&C1, M, N);
  armas_x_init(&tau0, imin(M, N), 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // C is first N columns of identity matrix
  armas_x_set_values(&C0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&C1, &C0);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_x_lqmult_work(&C0, ARMAS_RIGHT, &conf);
  armas_x_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  armas_x_lqfactor(&A0, &tau0, &W, &conf);

  // compute C0 = C0*Q.T*Q
  conf.error = 0;
  armas_x_lqmult(&C0, &A0, &tau0, &W, ARMAS_RIGHT|ARMAS_TRANS, &conf);
  armas_x_lqmult(&C0, &A0, &tau0, &W, ARMAS_RIGHT, &conf);

  n0 = rel_error((DTYPE *)0, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  printf("%s: %s C*Q.T*Q == C\n", PASS(isOK(n0, N)), blk);
  if (verbose > 0)
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));

  armas_x_release(&A0);
  armas_x_release(&C0);
  armas_x_release(&C1);
  armas_x_release(&W);
  armas_x_release(&tau0);
  return isOK(n0, N);
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

  if (! test_mult_identity(M, N, 0, verbose))
    fails += 1;
  if (! test_mult_identity(M, N, LB, verbose))
    fails += 1;

  if (! test_mult_left(M, N, 0, verbose))
    fails += 1;
  if (! test_mult_left(M, N, LB, verbose))
    fails += 1;

  if (! test_mult_right(M, N, 0, verbose))
    fails += 1;
  if (! test_mult_right(M, N, LB, verbose))
    fails += 1;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
