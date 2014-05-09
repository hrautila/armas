
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "ldl"


int test_factor(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1, W;
  armas_pivot_t P0, P1;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  double nrm;

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_pivot_init(&P0, N);
  armas_pivot_init(&P1, N);

  conf.lb = lb;
  wsize = armas_d_bkfactor_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, flags);
  armas_d_mcopy(&A1, &A0);

  conf.error = 0;
  conf.lb = 0; 
  armas_d_bkfactor(&A0, &W, &P0, flags, &conf);
  if (verbose > 1 && conf.error != 0)
    printf("1. error=%d\n", conf.error);

  conf.error = 0;
  conf.lb = lb;
  armas_d_bkfactor(&A1, &W, &P1, flags,  &conf);
  if (verbose > 1 && conf.error != 0)
    printf("1. error=%d\n", conf.error);

  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
  ok = isFINE(nrm, N*1e-12);
  
  printf("%s: unblk.LDL(A,%c) == blk.LDL(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("  ||unblk.LDL(A, '%c') - blk.LDL(A, '%c')||: %e [%ld]\n",
           uplo, uplo, nrm, (int64_t)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&W);
  armas_pivot_release(&P0);
  armas_pivot_release(&P1);

  return ok;
}


int test_solve(int M, int N, int lb, int verbose, int flags)
{
  armas_d_dense_t A0, A1;
  armas_d_dense_t B0, B1, X0, X1, X2, W;
  armas_pivot_t P0;
  armas_conf_t conf = *armas_conf_default();
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  int ok, wsize;
  double nrm;

  armas_d_init(&A0, N, N);
  armas_d_init(&A1, N, N);
  armas_d_init(&B0, N, M);
  armas_d_init(&X0, N, M);
  armas_pivot_init(&P0, N);

  // set source data
  armas_d_set_values(&A0, unitrand, flags);
  armas_d_mcopy(&A1, &A0);

  armas_d_set_values(&B0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&X0, &B0);

  conf.lb = lb;
  wsize = armas_d_bkfactor_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);
  armas_d_bkfactor(&A0, &W, &P0, flags, &conf);

  // solve
  armas_d_bksolve(&X0, &A0, &W, &P0, flags, &conf);
  
  // B0 = B0 - A*X0
  armas_d_mult_sym(&B0, &A1, &X0, -1.0, 1.0, ARMAS_LEFT|flags, &conf);
  nrm = armas_d_mnorm(&B0, ARMAS_NORM_ONE, &conf);

  ok = isFINE(nrm, N*1e-12);
  
  printf("%s: LDL(%c)  A*(A.-1*B) == B\n", PASS(ok), uplo);
  if (verbose > 0) {
    printf(" ||B - A*(A.-1*B)||: %e [%ld]\n", nrm, (int64_t)(nrm/DBL_EPSILON));
  }
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 657;
  int N = 657;
  int K = N;
  int LB = 40;
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
