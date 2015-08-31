
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-13
#endif

#define NAME "hessred"

int test_reduce(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1, tau0, tau1, W;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  int wchange = lb > 8 ? 2*M : 0;
  __Dtype n0, n1;

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&tau0, N, 1);
  matrix_init(&tau1, N, 1);

  conf.lb = lb;
  wsize = matrix_hessreduce_work(&A0, &conf);
  matrix_init(&W, wsize-wchange, 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  // unblocked reduction
  conf.lb = 0;
  matrix_hessreduce(&A0, &tau0, &W, &conf);

  // blocked reduction
  conf.lb = lb;
  matrix_hessreduce(&A1, &tau1, &W, &conf);


  n0 = rel_error((__Dtype *)0, &A0,   &A1,   ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  n1 = rel_error((__Dtype *)0, &tau0, &tau1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
  ok = isFINE(n0, N*__ERROR);

  printf("%s: unblk.Hess(A) == blk.Hess(A)\n", PASS(ok));
  if (verbose > 0) {
    printf("  || error.Hess ||: %e [%d]\n", n0, ndigits(n0));
    printf("  || error.tau  ||: %e [%d]\n", n1, ndigits(n1));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&tau0);
  matrix_release(&tau1);
  matrix_release(&W);

  return ok;
}


int test_mult(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1, B, tau0, W, Blow;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  int wchange = lb > 8 ? 2*M : 0;
  __Dtype nrm;

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&B, N, N);
  matrix_init(&tau0, N, 1);

  conf.lb = lb;
  // A is square; left and right work sizes are equal
  wsize = matrix_hessmult_work(&A0, ARMAS_LEFT, &conf);
  matrix_init(&W, wsize-wchange, 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  // reduce to Hessenberg matrix
  conf.lb = lb;
  matrix_hessreduce(&A0, &tau0, &W, &conf);

  // extract B = Hess(A)  
  matrix_mcopy(&B, &A0);
  matrix_submatrix(&Blow, &B, 1, 0, N-1, N-1);
  matrix_make_trm(&Blow, ARMAS_UPPER);

  // A = H*B*H.T; update B with H.T and H
  matrix_hessmult(&B, &A0, &tau0, &W, ARMAS_LEFT, &conf);
  matrix_hessmult(&B, &A0, &tau0, &W, ARMAS_RIGHT|ARMAS_TRANS, &conf);

  // B == A1?
  nrm = rel_error((__Dtype *)0, &B, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);
  printf("%s: Q*Hess(A)*Q.T == A\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&B);
  matrix_release(&tau0);
  matrix_release(&W);
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

  if (! test_reduce(M, N, LB, verbose) )
    fails++;
  if (! test_mult(M, N, LB, verbose) )
    fails++;
  
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
