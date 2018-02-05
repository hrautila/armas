
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "ldl"


int test_factor(int M, int N, int lb, int verbose, int flags)
{
  armas_x_dense_t A0, A1, W;
  armas_pivot_t P0, P1;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  DTYPE nrm;

  armas_x_init(&A0, N, N);
  armas_x_init(&A1, N, N);
  armas_pivot_init(&P0, N);
  armas_pivot_init(&P1, N);

  conf.lb = lb;
  wsize = armas_x_bkfactor_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, flags);
  armas_x_mcopy(&A1, &A0);

  conf.error = 0;
  conf.lb = 0; 
  armas_x_bkfactor(&A0, &W, &P0, flags, &conf);
  if (verbose > 1 && conf.error != 0)
    printf("1. error=%d\n", conf.error);

  conf.error = 0;
  conf.lb = lb;
  armas_x_bkfactor(&A1, &W, &P1, flags,  &conf);
  if (verbose > 1 && conf.error != 0)
    printf("1. error=%d\n", conf.error);

  nrm = rel_error((DTYPE *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);

  
  printf("%s: unblk.LDL(A,%c) == blk.LDL(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("  || error.LDL(A, '%c') ||: %e [%d]\n", uplo, nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&A1);
  armas_x_release(&W);
  armas_pivot_release(&P0);
  armas_pivot_release(&P1);

  return ok;
}


int test_solve(int M, int N, int lb, int verbose, int flags)
{
  armas_x_dense_t A0, A1;
  armas_x_dense_t B0, X0, W;
  armas_pivot_t P0;
  armas_conf_t conf = *armas_conf_default();
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  int ok, wsize;
  DTYPE nrm, nrm_A;

  armas_x_init(&A0, N, N);
  armas_x_init(&A1, N, N);
  armas_x_init(&B0, N, M);
  armas_x_init(&X0, N, M);
  armas_pivot_init(&P0, N);

  // set source data
  armas_x_set_values(&A0, unitrand, flags);
  armas_x_mcopy(&A1, &A0);

  armas_x_set_values(&B0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&X0, &B0);
  nrm_A = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf);

  conf.lb = lb;
  wsize = armas_x_bkfactor_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);
  armas_x_bkfactor(&A0, &W, &P0, flags, &conf);

  // solve
  armas_x_bksolve(&X0, &A0, &W, &P0, flags, &conf);
  
  // B0 = B0 - A*X0
  armas_x_mult_sym(1.0, &B0, -1.0, &A1, &X0, ARMAS_LEFT|flags, &conf);
  nrm = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf);
  nrm /= nrm_A;

#if FLOAT32
  ok = isFINE(nrm, N*8e-6);
#else
  ok = isFINE(nrm, N*1e-11);
#endif
  
  printf("%s: LDL(%c)  A*(A.-1*B) == B\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf(" || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }
  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int M = 657;
  int N = 657;
  int LB = 40;
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
  if ( ! test_factor(M, N, LB, verbose, ARMAS_LOWER) )
    fails++;
  if ( ! test_factor(M, N, LB, verbose, ARMAS_UPPER) )
    fails++;
  if ( ! test_solve(M, N, LB, verbose, ARMAS_LOWER) )
    fails++;
  if ( ! test_solve(M, N, LB, verbose, ARMAS_UPPER) )
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
