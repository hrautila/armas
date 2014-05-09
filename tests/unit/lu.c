
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "test"

extern
int netlib_dgetrf(armas_d_dense_t *A, armas_pivot_t *P);

extern
int netlib_dgetrs(armas_d_dense_t *B, armas_d_dense_t *A, armas_pivot_t *P, int flags);

int test_solve(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1;
  armas_d_dense_t B0, X0;
  armas_pivot_t P0;
  armas_conf_t conf = *armas_conf_default();
  char *blk = lb == 0 ? "unblk" : "blk";
  int ok;
  double nrm;

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&B0, N, M);
  armas_d_init(&X0, N, M);
  armas_pivot_init(&P0, N);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  armas_d_set_values(&B0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&X0, &B0);

  conf.lb = lb;
  armas_d_lufactor(&A0, &P0, &conf);

  // solve
  armas_d_lusolve(&X0, &A0, &P0, ARMAS_NONE, &conf);

  // B0 = B0 - A*X0
  armas_d_mult(&B0, &A1, &X0, -1.0, 1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&B0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-12);

  printf("%s: A*(%s.LU(A).1*B) == B\n", PASS(ok), blk);
  if (verbose > 0) {
    printf("  ||B - A*(A.-1*B)||: %e [%ld]\n", nrm, (int64_t)(nrm/DBL_EPSILON));
  }
  return ok;
}

int test_factor(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, A2;
  armas_pivot_t P0, P1, P2;
  armas_pivot_t *P = (armas_pivot_t *)0;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  double nrm;

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_pivot_init(&P0, N);
  armas_pivot_init(&P1, N);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  //armas_d_lufactor(&A0, &P0, &conf);
  conf.lb = 0;
  armas_d_lufactor(&A0, &P0, &conf);
  conf.lb = lb;
  armas_d_lufactor(&A1, &P1,  &conf);

  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-12);

  printf("%s: unblk.LU(A) == blk.LU(A)\n", PASS(ok));
  if (verbose > 0) {
    printf("  ||unblk.LU(A) - blk.LU(A)||: %e [%ld]\n", nrm, (int64_t)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_pivot_release(&P0);
  armas_pivot_release(&P1);
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 789;
  int N = 711;
  int K = N;
  int LB = 36;
  int ok = 0;
  int nproc = 1;
  int verbose = 1;
  int testno = 0;

  while ((opt = getopt(argc, argv, "T:P:v")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    case 'T':
      testno = atoi(optarg);
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
