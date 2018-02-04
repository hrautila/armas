
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
#define MAX_ERROR 1e-12
#endif

int test_solve(int M, int N, int lb, int verbose)
{
  armas_x_dense_t A0, A1;
  armas_x_dense_t B0, X0;
  armas_pivot_t P0;
  armas_conf_t conf = *armas_conf_default();
  char *blk = lb == 0 ? "unblk" : "blk";
  int ok;
  DTYPE nrm, nrm0;

  armas_x_init(&A0, N, N);
  armas_x_init(&A1, N, N);
  armas_x_init(&B0, N, M);
  armas_x_init(&X0, N, M);
  armas_pivot_init(&P0, N);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);

  armas_x_set_values(&B0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&X0, &B0);
  nrm0 = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf);

  conf.lb = lb;
  armas_x_lufactor(&A0, &P0, &conf);

  // solve
  armas_x_lusolve(&X0, &A0, &P0, ARMAS_NONE, &conf);

  // B0 = B0 - A*X0
  armas_x_mult(1.0, &B0, -1.0, &A1, &X0, ARMAS_NONE, &conf);
  nrm = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf) / nrm0;

  ok = isFINE(nrm, N*MAX_ERROR);

  printf("%s: A*(%s.LU(A).1*B) == B\n", PASS(ok), blk);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  return ok;
}

int test_factor(int M, int N, int lb, int verbose)
{
  armas_x_dense_t A0, A1;
  armas_pivot_t P0, P1;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  DTYPE nrm;

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_pivot_init(&P0, N);
  armas_pivot_init(&P1, N);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);

  //armas_x_lufactor(&A0, &P0, &conf);
  conf.lb = 0;
  armas_x_lufactor(&A0, &P0, &conf);
  conf.lb = lb;
  armas_x_lufactor(&A1, &P1,  &conf);

  nrm = rel_error((DTYPE *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);

  printf("%s: unblk.LU(A) == blk.LU(A)\n", PASS(ok));
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&A1);
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
