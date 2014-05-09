
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "chol"

int test_solve(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1;
  armas_d_dense_t B0, X0;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  double nrm;
  char *uplo = flags & ARMAS_UPPER ? "Upper" : "Lower";

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&B0, N, M);
  armas_d_init(&X0, N, M);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_d_mcopy(&A0, &A1);

  armas_d_set_values(&B0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&X0, &B0);

  conf.lb = lb;
  armas_d_cholfactor(&A0, flags, &conf);

  // solve
  armas_d_cholsolve(&X0, &A0, flags, &conf);

  // X0 = A*X0 - B0
  armas_d_mult(&B0, &A1, &X0, -1.0, 1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&B0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-9);

  printf("%s: A*(CHOLsolve(A, B, %s)) == B\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf("   %s.||B - A*(A.-1*B)||: %e [%ld]\n",
           uplo, nrm, (int64_t)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&B0);
  armas_d_release(&X0);

  return ok;
}

int test_factor(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1, A2;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  double nrm;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  // A = A*A.T; positive semi-definite
  armas_d_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_d_mcopy(&A0, &A1);

  conf.lb = 0; 
  armas_d_cholfactor(&A0, flags, &conf);
  conf.lb = lb;
  armas_d_cholfactor(&A1, flags,  &conf);

  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-9);

  printf("%s: unblk.CHOL(A,%c) == blk.CHOL(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("   ||unblk.CHOL(A,%c) - blk.CHOL(A,%c)||: %e [%ld]\n",
           uplo, uplo, nrm, (int64_t)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 511;
  int N = 779;
  int K = N;
  int LB = 36;
  int ok = 0;
  int nproc = 1;
  int verbose = 1;
  int flags = ARMAS_LOWER;

  while ((opt = getopt(argc, argv, "P:v")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    case 'v':
      verbose += 1;
      break;
    default:
      fprintf(stderr, "usage: %s [-v]  [M N LB]\n", NAME);
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

  if (! test_factor(M, N, LB, verbose, ARMAS_LOWER))
    fails++;
  if (! test_factor(M, N, LB, verbose, ARMAS_UPPER))
    fails++;

  if (! test_solve(M, N, LB, verbose, ARMAS_LOWER))
    fails++;

  if (! test_solve(M, N, LB, verbose, ARMAS_UPPER))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
