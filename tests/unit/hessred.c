
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "hessred"

int test_reduce(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, tau0, tau1, W, tmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  int wchange = lb > 8 ? 2*M : 0;
  double nrm;

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&tau0, N, 1);
  armas_d_init(&tau1, N, 1);

  conf.lb = lb;
  wsize = armas_d_hessreduce_work(&A0, &conf);
  armas_d_init(&W, wsize-wchange, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  // unblocked reduction
  conf.lb = 0;
  armas_d_hessreduce(&A0, &tau0, &W, &conf);

  // blocked reduction
  conf.lb = lb;
  armas_d_hessreduce(&A1, &tau1, &W, &conf);

  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);

  ok = isFINE(nrm, N*1e-12);
  printf("%s: unblk.Hess(A) == blk.Hess(A)\n", PASS(ok));
  if (verbose > 0) {
    printf("  ||unblk.Hess(A) - blk.Hess(A)||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&tau0);
  armas_d_release(&tau1);
  armas_d_release(&W);

  return ok;
}


int test_mult(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, B, tau0, W, Blow;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize, err;
  int wchange = lb > 8 ? 2*M : 0;
  double nrm;

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&B, N, N);
  armas_d_init(&tau0, N, 1);

  conf.lb = lb;
  // A is square; left and right work sizes are equal
  wsize = armas_d_hessmult_work(&A0, ARMAS_LEFT, &conf);
  armas_d_init(&W, wsize-wchange, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  // reduce to Hessenberg matrix
  conf.lb = lb;
  armas_d_hessreduce(&A0, &tau0, &W, &conf);

  // extract B = Hess(A)  
  armas_d_mcopy(&B, &A0);
  armas_d_submatrix(&Blow, &B, 1, 0, N-1, N-1);
  armas_d_make_trm(&Blow, ARMAS_UPPER);

  // A = H*B*H.T; update B with H.T and H
  armas_d_hessmult(&B, &A0, &tau0, &W, ARMAS_LEFT, &conf);
  armas_d_hessmult(&B, &A0, &tau0, &W, ARMAS_RIGHT|ARMAS_TRANS, &conf);

  // B == A1?
  armas_d_scale_plus(&B, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  ok = isOK(nrm, N);
  printf("%s: Q*Hess(A)*Q.T == A\n", PASS(ok));
  if (verbose > 0) {
    printf("  ||A - Q*Hess(A)*Q.T||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&B);
  armas_d_release(&tau0);
  armas_d_release(&W);
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

  if (! test_reduce(M, N, LB, verbose) )
    fails++;
  if (! test_mult(M, N, LB, verbose) )
    fails++;
  
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
