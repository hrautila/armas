
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "trdreduce"

int test_reduce(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1, tau0, tau1, W, tmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  char uplo = flags & ARMAS_LOWER ? 'L' : 'U';
  double nrm;

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&tau0, N, 1);
  armas_d_init(&tau1, N, 1);

  conf.lb = lb;
  wsize = armas_d_trdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, flags);
  armas_d_mcopy(&A1, &A0);

  conf.lb = 0;
  armas_d_trdreduce(&A0, &tau0, &W, flags, &conf);

  conf.lb = lb;
  armas_d_trdreduce(&A1, &tau1, &W, flags, &conf);

  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  armas_d_axpy(&tau0, &tau1, -1.0, &conf);
  nrm = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-12);

  printf("%s: unblk.TRD(A,%c) == blk.TRD(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("  ||unblk.TRD(A,%c) - blk.TRD(A,%c)||: %e [%d]\n",
           uplo, uplo, nrm, (int)(nrm/DBL_EPSILON));

    nrm = armas_d_nrm2(&tau0, &conf);
    printf("  ||unblk.tau - blk.tau||: %e\n", nrm);
  }
  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&tau0);
  armas_d_release(&tau1);
  armas_d_release(&W);
  return ok;
}


// compute ||A - Q*T*Q.T||
int test_mult_trd(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1, tau0,  W, tmp, T0, T1, Qh, e1, e2, d1, d2;
  armas_d_dense_t D, E;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize, err;
  double nrm;
  int tN = M < N ? M : N;
  char *uplo = flags & ARMAS_UPPER ? "UPPER" : "LOWER";

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&T0, N, N);
  armas_d_init(&T1, N, N);
  armas_d_init(&tau0, N, 1);

  conf.lb = lb;
  wsize = armas_d_trdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, flags);
  armas_d_mcopy(&A1, &A0);

  conf.lb = lb;
  armas_d_trdreduce(&A0, &tau0, &W, flags, &conf);

  // make tridiagonal matrix T0
  armas_d_diag(&d1, &A0, 0);
  armas_d_diag(&d2, &T0, 0);
  armas_d_mcopy(&d2, &d1);
  if (flags & ARMAS_UPPER) {
    armas_d_diag(&e1, &A0, 1);
  } else {
    armas_d_diag(&e1, &A0, -1);
  }
  // copy off-diagonals
  armas_d_diag(&e2, &T0, 1);
  armas_d_mcopy(&e2, &e1);
  armas_d_diag(&e2, &T0, -1);
  armas_d_mcopy(&e2, &e1);

  // compute Q*T*Q.T
  armas_d_trdmult(&T0, &A0, &tau0, &W, flags|ARMAS_LEFT, &conf);
  armas_d_trdmult(&T0, &A0, &tau0, &W, flags|ARMAS_RIGHT|ARMAS_TRANS, &conf);

  // make result triangular (original matrix)
  armas_d_make_trm(&T0, flags);
  armas_d_scale_plus(&T0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);

  nrm = armas_d_mnorm(&T0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-12);
  printf("%s: [%s] Q*T*Q.T == A\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("  ||A - Q*T*Q.T||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }
  return ok;
}


// compute ||T - Q.T*A*Q||
int test_mult_a(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1, tau0,  W, tmp, T0, T1, Qh, e1, e2, d1, d2;
  armas_d_dense_t D, E;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize, err;
  double nrm;
  int tN = M < N ? M : N;
  char *uplo = flags & ARMAS_UPPER ? "UPPER" : "LOWER";

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&T0, N, N);
  armas_d_init(&T1, N, N);
  armas_d_init(&tau0, N, 1);

  conf.lb = lb;
  wsize = armas_d_trdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data; make it full symmetric
  armas_d_set_values(&A0, unitrand, ARMAS_SYMM);
  armas_d_mcopy(&A1, &A0);

  conf.lb = lb;
  armas_d_trdreduce(&A0, &tau0, &W, flags, &conf);

  // make tridiagonal matrix T0
  armas_d_diag(&d1, &A0, 0);
  armas_d_diag(&d2, &T0, 0);
  armas_d_mcopy(&d2, &d1);
  if (flags & ARMAS_UPPER) {
    armas_d_diag(&e1, &A0, 1);
  } else {
    armas_d_diag(&e1, &A0, -1);
  }
  // copy off-diagonals
  armas_d_diag(&e2, &T0, 1);
  armas_d_mcopy(&e2, &e1);
  armas_d_diag(&e2, &T0, -1);
  armas_d_mcopy(&e2, &e1);

  // compute Q.T*A*Q
  armas_d_trdmult(&A1, &A0, &tau0, &W, flags|ARMAS_LEFT|ARMAS_TRANS, &conf);
  armas_d_trdmult(&A1, &A0, &tau0, &W, flags|ARMAS_RIGHT, &conf);

  armas_d_scale_plus(&T0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&T0, ARMAS_NORM_ONE, &conf);

  ok = isFINE(nrm, N*1e-12);
  printf("%s: [%s] Q.T*A*Q == T\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("  ||T - Q.T*A*Q||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }
  return ok;
}


int test_build(int M, int N, int lb, int K, int verbose, int flags)
{
  armas_d_dense_t A0, tauq0, d0, W, tmp, QQt;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  double nrm;
  int tN = M < N ? M : N;
  char *uplo = flags & ARMAS_UPPER ? "UPPER" : "LOWER";

  armas_d_init(&A0, N, N);
  armas_d_init(&tauq0, tN, 1);
  //------------------------------------------------------

  conf.lb = lb;
  wsize = armas_d_trdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, flags);
  // reduce to tridiagonal matrix
  conf.lb = lb;
  armas_d_trdreduce(&A0, &tauq0, &W, flags, &conf);
  // ----------------------------------------------------------------
  // Q-matrix
  armas_d_trdbuild(&A0, &tauq0, &W, K, flags, &conf);
  armas_d_init(&QQt, N, N);
  armas_d_mult(&QQt, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_d_diag(&d0, &QQt, 0);
  armas_d_madd(&d0, -1.0, ARMAS_NONE);

  nrm = armas_d_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
  ok = isFINE(nrm, N*1e-12);
  printf("%s: [%s] I == Q*Q.T\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("  ||I - Q*Q.T||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }
  //------------------------------------------------------
  armas_d_release(&A0);
  armas_d_release(&tauq0);
  armas_d_release(&W);
  armas_d_release(&QQt);
  return ok;
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
  if (! test_reduce(N, N, LB, verbose, ARMAS_LOWER))
    fails++;
  if (! test_reduce(N, N, LB, verbose, ARMAS_UPPER))
    fails++;
  if (! test_mult_trd(N, N, LB, verbose, ARMAS_LOWER))
    fails++;
  if (! test_mult_trd(N, N, LB, verbose, ARMAS_UPPER))
    fails++;
  if (! test_mult_a(N, N, LB, verbose, ARMAS_LOWER))
    fails++;
  if (! test_mult_a(N, N, LB, verbose, ARMAS_UPPER))
    fails++;
  if (! test_build(N, N, LB, N/2, verbose, ARMAS_LOWER))
    fails++;
  if (! test_build(N, N, LB, N/2, verbose, ARMAS_UPPER))
    fails++;
  
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
