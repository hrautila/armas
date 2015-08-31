
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR 6e-6
#else
#define __ERROR 1e-12
#endif

#define NAME "trdreduce"

int test_reduce(int M, int N, int lb, int verbose, int flags)
{
  __Matrix A0, A1, tau0, tau1, W;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  char uplo = flags & ARMAS_LOWER ? 'L' : 'U';
  __Dtype n0, n1;

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&tau0, N, 1);
  matrix_init(&tau1, N, 1);

  conf.lb = lb;
  wsize = matrix_trdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, flags);
  matrix_mcopy(&A1, &A0);

  conf.lb = 0;
  matrix_trdreduce(&A0, &tau0, &W, flags, &conf);

  conf.lb = lb;
  matrix_trdreduce(&A1, &tau1, &W, flags, &conf);


  n0 = rel_error((__Dtype *)0, &A0,   &A1,   ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  n1 = rel_error((__Dtype *)0, &tau0, &tau1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
  ok = isFINE(n0, N*__ERROR);

  printf("%s: unblk.TRD(A,%c) == blk.TRD(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("  || error.TRD(A,%c)||: %e [%d]\n", uplo, n0, ndigits(n0));

    printf("  || error.tau||      : %e [%d]\n", n1, ndigits(n1));
  }
  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&tau0);
  matrix_release(&tau1);
  matrix_release(&W);
  return ok;
}


// compute ||A - Q*T*Q.T||
int test_mult_trd(int M, int N, int lb, int verbose, int flags)
{
  __Matrix A0, A1, tau0,  W, T0, T1, e1, e2, d1, d2;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  char *uplo = flags & ARMAS_UPPER ? "UPPER" : "LOWER";

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&T0, N, N);
  matrix_init(&T1, N, N);
  matrix_init(&tau0, N, 1);

  conf.lb = lb;
  wsize = matrix_trdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, flags);
  matrix_mcopy(&A1, &A0);

  conf.lb = lb;
  matrix_trdreduce(&A0, &tau0, &W, flags, &conf);

  // make tridiagonal matrix T0
  matrix_diag(&d1, &A0, 0);
  matrix_diag(&d2, &T0, 0);
  matrix_mcopy(&d2, &d1);
  if (flags & ARMAS_UPPER) {
    matrix_diag(&e1, &A0, 1);
  } else {
    matrix_diag(&e1, &A0, -1);
  }
  // copy off-diagonals
  matrix_diag(&e2, &T0, 1);
  matrix_mcopy(&e2, &e1);
  matrix_diag(&e2, &T0, -1);
  matrix_mcopy(&e2, &e1);

  // compute Q*T*Q.T
  matrix_trdmult(&T0, &A0, &tau0, &W, flags|ARMAS_LEFT, &conf);
  matrix_trdmult(&T0, &A0, &tau0, &W, flags|ARMAS_RIGHT|ARMAS_TRANS, &conf);

  // make result triangular (original matrix)
  matrix_make_trm(&T0, flags);
  nrm = rel_error((__Dtype *)0, &T0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: [%s] Q*T*Q.T == A\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  return ok;
}


// compute ||T - Q.T*A*Q||
int test_mult_a(int M, int N, int lb, int verbose, int flags)
{
  __Matrix A0, A1, tau0,  W, T0, T1, e1, e2, d1, d2;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  char *uplo = flags & ARMAS_UPPER ? "UPPER" : "LOWER";

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&T0, N, N);
  matrix_init(&T1, N, N);
  matrix_init(&tau0, N, 1);

  conf.lb = lb;
  wsize = matrix_trdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data; make it full symmetric
  matrix_set_values(&A0, unitrand, ARMAS_SYMM);
  matrix_mcopy(&A1, &A0);

  conf.lb = lb;
  matrix_trdreduce(&A0, &tau0, &W, flags, &conf);

  // make tridiagonal matrix T0
  matrix_diag(&d1, &A0, 0);
  matrix_diag(&d2, &T0, 0);
  matrix_mcopy(&d2, &d1);
  if (flags & ARMAS_UPPER) {
    matrix_diag(&e1, &A0, 1);
  } else {
    matrix_diag(&e1, &A0, -1);
  }
  // copy off-diagonals
  matrix_diag(&e2, &T0, 1);
  matrix_mcopy(&e2, &e1);
  matrix_diag(&e2, &T0, -1);
  matrix_mcopy(&e2, &e1);

  // compute Q.T*A*Q
  matrix_trdmult(&A1, &A0, &tau0, &W, flags|ARMAS_LEFT|ARMAS_TRANS, &conf);
  matrix_trdmult(&A1, &A0, &tau0, &W, flags|ARMAS_RIGHT, &conf);

  nrm = rel_error((__Dtype *)0, &T0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: [%s] Q.T*A*Q == T\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  return ok;
}


int test_build(int M, int N, int lb, int K, int verbose, int flags)
{
  __Matrix A0, tauq0, d0, W, QQt;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  int tN = M < N ? M : N;
  char *uplo = flags & ARMAS_UPPER ? "UPPER" : "LOWER";

  matrix_init(&A0, N, N);
  matrix_init(&tauq0, tN, 1);
  //------------------------------------------------------

  conf.lb = lb;
  wsize = matrix_trdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, flags);
  // reduce to tridiagonal matrix
  conf.lb = lb;
  matrix_trdreduce(&A0, &tauq0, &W, flags, &conf);
  // ----------------------------------------------------------------
  // Q-matrix
  matrix_trdbuild(&A0, &tauq0, &W, K, flags, &conf);
  matrix_init(&QQt, N, N);
  matrix_mult(&QQt, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  matrix_diag(&d0, &QQt, 0);
  matrix_madd(&d0, -1.0, ARMAS_NONE);

  nrm = matrix_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: [%s] I == Q*Q.T\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  //------------------------------------------------------
  matrix_release(&A0);
  matrix_release(&tauq0);
  matrix_release(&W);
  matrix_release(&QQt);
  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  //int M = 787;
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
    
  if (optind < argc-1) {
    N = atoi(argv[optind]);
    LB = atoi(argv[optind+1]);
  } else if (optind < argc) {
    N = atoi(argv[optind]);
    LB = 0;
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
