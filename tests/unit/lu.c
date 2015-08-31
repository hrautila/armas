
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "test"
#if FLOAT32
#define MAX_ERROR 8e-6
#else
#define MAX_ERRRO 1e-12
#endif

int test_solve(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1;
  __Matrix B0, X0;
  armas_pivot_t P0;
  armas_conf_t conf = *armas_conf_default();
  char *blk = lb == 0 ? "unblk" : "blk";
  int ok;
  __Dtype nrm, nrm0;

  matrix_init(&A0, N, N);
  matrix_init(&A1, N, N);
  matrix_init(&B0, N, M);
  matrix_init(&X0, N, M);
  armas_pivot_init(&P0, N);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  matrix_set_values(&B0, unitrand, ARMAS_ANY);
  matrix_mcopy(&X0, &B0);
  nrm0 = matrix_mnorm(&B0, ARMAS_NORM_ONE, &conf);

  conf.lb = lb;
  matrix_lufactor(&A0, &P0, &conf);

  // solve
  matrix_lusolve(&X0, &A0, &P0, ARMAS_NONE, &conf);

  // B0 = B0 - A*X0
  matrix_mult(&B0, &A1, &X0, -1.0, 1.0, ARMAS_NONE, &conf);
  nrm = matrix_mnorm(&B0, ARMAS_NORM_ONE, &conf) / nrm0;

  ok = isFINE(nrm, N*MAX_ERROR);

  printf("%s: A*(%s.LU(A).1*B) == B\n", PASS(ok), blk);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  return ok;
}

int test_factor(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1;
  armas_pivot_t P0, P1;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  __Dtype nrm;

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  armas_pivot_init(&P0, N);
  armas_pivot_init(&P1, N);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  //matrix_lufactor(&A0, &P0, &conf);
  conf.lb = 0;
  matrix_lufactor(&A0, &P0, &conf);
  conf.lb = lb;
  matrix_lufactor(&A1, &P1,  &conf);

  nrm = rel_error((__Dtype *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);

  printf("%s: unblk.LU(A) == blk.LU(A)\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  armas_pivot_release(&P0);
  armas_pivot_release(&P1);
  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int M = 789;
  int N = 711;
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

  if (! test_factor(M, N, LB, verbose))
    fails++;

  // unblocked
  if (! test_solve(M, N, 0, verbose))
    fails++;

  // blocked
  if (! test_solve(M, N, LB, verbose))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
