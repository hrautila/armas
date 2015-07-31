
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "qrmult"

/*  ------------------------------------------------------------------------
 *  Test 2: I - Q.T*Q = 0.0
 *    OK: ||I - Q.T*Q||_1 ~~ n*eps
 *  
 */
int test_mult_identity(int M, int N, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  armas_d_dense_t A0, C, C1, tau0, W, D;
  int wsize, ok;
  double n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  armas_d_init(&A0, M, N);
  armas_d_init(&C, M, N);
  armas_d_init(&C1, M, N);
  armas_d_init(&tau0, imin(M, N), 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // C is first N columns of identity matrix
  armas_d_set_values(&C, zero, ARMAS_ANY);
  armas_d_diag(&D, &C, 0);
  armas_d_add(&D, 1.0, &conf);
  armas_d_mcopy(&C1, &C);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_d_qrmult_work(&C, ARMAS_LEFT, &conf);
  armas_d_init(&W, wsize, 1);

  // factorize
  armas_d_qrfactor(&A0, &tau0, &W, &conf);

  armas_d_qrmult(&C, &A0, &tau0, &W, ARMAS_LEFT, &conf);
  armas_d_qrmult(&C, &A0, &tau0, &W, ARMAS_LEFT|ARMAS_TRANS, &conf);

  n0 = rel_error((double *)0, &C, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(n0, N);

  printf("%s: %s Q.T*Q == I\n", PASS(ok), blk);
  if (verbose > 0) {
    printf("  || rel_error ||_1: %e [%d]\n", n0, ndigits(n0));
  }
  armas_d_release(&A0);
  armas_d_release(&C);
  armas_d_release(&C1);
  armas_d_release(&tau0);
  return ok;
}

/*  ----------------------------------------------------------------------------
 *  Test 3: C == Q.T*Q*C
 *    OK: ||C - Q.T*Q*C||_1 ~~ n*eps
 */
int test_mult_left(int M, int N, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  armas_d_dense_t A0, C1, C0, tau0, W;
  int wsize, ok;
  double n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  armas_d_init(&A0, M, N);
  armas_d_init(&C0, M, N);
  armas_d_init(&C1, M, N);
  armas_d_init(&tau0, imin(M, N), 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // C is first N columns of identity matrix
  armas_d_set_values(&C0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&C1, &C0);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_d_qrmult_work(&C0, ARMAS_LEFT, &conf);
  armas_d_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  armas_d_qrfactor(&A0, &tau0, &W, &conf);

  // compute C0 = Q.T*Q*C0
  armas_d_qrmult(&C0, &A0, &tau0, &W, ARMAS_LEFT, &conf);
  armas_d_qrmult(&C0, &A0, &tau0, &W, ARMAS_LEFT|ARMAS_TRANS, &conf);

  n0 = rel_error((double *)0, &C1, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok =isOK(n0, N);

  printf("%s: %s Q.T*Q*C == C\n", PASS(ok), blk);
  if (verbose > 0) {
    printf("  || rel_error ||_1: %e [%d]\n", n0, ndigits(n0));
  }

  armas_d_release(&A0);
  armas_d_release(&C0);
  armas_d_release(&C1);
  armas_d_release(&tau0);
  return ok; 
}


/*  -----------------------------------------------------------------------------------
 *  Test 4: C == C*Q.T*Q
 *    OK: ||C - C*Q.T*Q||_1 ~~ n*eps
 */
int test_mult_right(int M, int N, int lb, int verbose)
{
  char *blk = lb > 0 ? "  blk" : "unblk";
  armas_d_dense_t A0, C1, C0, tau0, W;
  int wsize, ok;
  double n0, n1;
  armas_conf_t conf = *armas_conf_default();
  
  armas_d_init(&A0, M, N);
  armas_d_init(&C0, N, M);
  armas_d_init(&C1, N, M);
  armas_d_init(&tau0, imin(M, N), 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // C is first N columns of identity matrix
  armas_d_set_values(&C0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&C1, &C0);

  // allocate workspace according the blocked multiplication
  conf.lb = lb;
  wsize = armas_d_qrmult_work(&C0, ARMAS_RIGHT, &conf);
  armas_d_init(&W, wsize, 1);

  // factorize
  conf.lb = lb;
  armas_d_qrfactor(&A0, &tau0, &W, &conf);

  // compute C0 = C0*Q.T*Q
  int err;
  err = armas_d_qrmult(&C0, &A0, &tau0, &W, ARMAS_RIGHT|ARMAS_TRANS, &conf);
  if (verbose > 2 && N <= 10) {
    printf("err = %d, error = %d\n", err, conf.error);
    printf("C0*Q.T:\n"); armas_d_printf(stdout, "%9.2e", &C1);
  }
  armas_d_qrmult(&C0, &A0, &tau0, &W, ARMAS_RIGHT, &conf);

  if (verbose > 1 && N <= 10) {
    printf("C0:\n"); armas_d_printf(stdout, "%9.2e", &C0);
    printf("C0*Q.T*Q:\n"); armas_d_printf(stdout, "%9.2e", &C1);
  }

  n0 = rel_error((double *)0, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(n0, N);

  printf("%s: %s C*Q.T*Q == C\n", PASS(ok), blk);
  if (verbose > 0)
    printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));

  armas_d_release(&A0);
  armas_d_release(&C0);
  armas_d_release(&C1);
  armas_d_release(&tau0);
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 787;
  int N = 741;
  int K = N;
  int LB = 36;
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
