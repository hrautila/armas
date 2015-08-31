
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#if FLOAT32
#define __ERROR 8e-3
#else
#define __ERROR 1e-8
#endif

#define NAME "chol"

int test_solve(int M, int N, int lb, int verbose, int flags)
{
  __Matrix A0, A1;
  __Matrix B0, X0;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  __Dtype nrm, nrm0;
  char *uplo = flags & ARMAS_UPPER ? "Upper" : "Lower";
  char *blk = lb != 0 ? "  blk" : "unblk";

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&B0, N, M);
  matrix_init(&X0, N, M);

  // set source data (A = A*A.T)
  matrix_set_values(&A0, zeromean, ARMAS_ANY);
  matrix_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  matrix_mcopy(&A0, &A1);

  matrix_set_values(&B0, unitrand, ARMAS_ANY);
  nrm0 = matrix_mnorm(&B0, ARMAS_NORM_ONE, &conf);
  matrix_mcopy(&X0, &B0);

  conf.lb = lb;
  matrix_cholfactor(&A0, flags, &conf);

  // solve
  matrix_cholsolve(&X0, &A0, flags, &conf);

  // X0 = A*X0 - B0
  matrix_mult(&B0, &A1, &X0, -1.0, 1.0, ARMAS_NONE, &conf);
  nrm = matrix_mnorm(&B0, ARMAS_NORM_ONE, &conf) / nrm0;
  ok = isFINE(nrm, N*__ERROR);

  printf("%s: A*(%s.CHOLsolve(A, B, %s)) == B\n", PASS(ok), blk, uplo);
  if (verbose > 0) {
    printf("   || rel error ||: %e [%d]\n",  nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&B0);
  matrix_release(&X0);

  return ok;
}

int test_factor(int M, int N, int lb, int verbose, int flags)
{
  __Matrix A0, A1;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  __Dtype nrm;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  // A = A*A.T; positive semi-definite
  matrix_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  matrix_mcopy(&A0, &A1);

  conf.lb = 0; 
  matrix_cholfactor(&A0, flags, &conf);
  conf.lb = lb;
  matrix_cholfactor(&A1, flags,  &conf);

  nrm = rel_error((__Dtype *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);

  printf("%s: unblk.CHOL(A,%c) == blk.CHOL(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("   || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int M = 511;
  int N = 779;
  int LB = 36;
  int verbose = 1;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
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

  if (! test_solve(M, N, 0, verbose, ARMAS_LOWER))
    fails++;

  if (! test_solve(M, N, 0, verbose, ARMAS_UPPER))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
