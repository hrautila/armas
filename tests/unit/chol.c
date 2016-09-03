
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
  armas_x_dense_t A0, A1;
  armas_x_dense_t B0, X0;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  DTYPE nrm, nrm0;
  char *uplo = flags & ARMAS_UPPER ? "Upper" : "Lower";
  char *blk = lb != 0 ? "  blk" : "unblk";

  armas_x_init(&A0, N, N);
  armas_x_init(&A1, N, N);
  armas_x_init(&B0, N, M);
  armas_x_init(&X0, N, M);

  // set source data (A = A*A.T)
  armas_x_set_values(&A0, zeromean, ARMAS_ANY);
  armas_x_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_x_mcopy(&A0, &A1);

  armas_x_set_values(&B0, unitrand, ARMAS_ANY);
  nrm0 = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf);
  armas_x_mcopy(&X0, &B0);

  conf.lb = lb;
  armas_x_cholfactor(&A0, ARMAS_NULL, ARMAS_NOPIVOT, flags, &conf);

  // solve
  armas_x_cholsolve(&X0, &A0, ARMAS_NOPIVOT, flags, &conf);

  // X0 = A*X0 - B0
  armas_x_mult(&B0, &A1, &X0, -1.0, 1.0, ARMAS_NONE, &conf);
  nrm = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &conf) / nrm0;
  ok = isFINE(nrm, N*__ERROR);

  printf("%s: A*(%s.CHOLsolve(A, B, %s)) == B\n", PASS(ok), blk, uplo);
  if (verbose > 0) {
    printf("   || rel error ||: %e [%d]\n",  nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&A1);
  armas_x_release(&B0);
  armas_x_release(&X0);

  return ok;
}

int test_factor(int M, int N, int lb, int verbose, int flags)
{
  armas_x_dense_t A0, A1;
  armas_conf_t conf = *armas_conf_default();
  int ok;
  DTYPE nrm;
  char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
  armas_x_init(&A0, N, N);
  armas_x_init(&A1, N, N);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  // A = A*A.T; positive semi-definite
  armas_x_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_x_mcopy(&A0, &A1);

  conf.lb = 0; 
  armas_x_cholfactor(&A0, ARMAS_NULL, ARMAS_NOPIVOT, flags, &conf);
  conf.lb = lb;
  armas_x_cholfactor(&A1, ARMAS_NULL, ARMAS_NOPIVOT, flags,  &conf);

  nrm = rel_error((DTYPE *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isOK(nrm, N);

  printf("%s: unblk.CHOL(A,%c) == blk.CHOL(A,%c)\n", PASS(ok), uplo, uplo);
  if (verbose > 0) {
    printf("   || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&A1);
  return ok;
}

int test_cholpv(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, C, D, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P;
    DTYPE n0, n1;
    int e, ok, flags1, flags2, pflgs;
    char *fact = flags & ARMAS_LOWER ? "P^T*(LL^T)*P" : "P^T*(U^TU)*P";
    char *blk = lb == 0 ? "unblk" : "  blk";
    pflgs = flags & ARMAS_LOWER ? ARMAS_PIVOT_LOWER : ARMAS_PIVOT_UPPER;
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&W, N, 1);
    armas_x_diag(&D, &C, 0);
    armas_x_madd(&D, 1.0, 0);
    armas_pivot_init(&P, N);
    
    armas_x_set_values(&A0, unitrand, 0);
    armas_x_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
    armas_x_make_trm(&A1, flags);
    armas_x_mcopy(&A0, &A1);
    if (N < 10) {
        printf("A:\n"); armas_x_printf(stdout, "%6.3f", &A0);
    }

    conf.lb = lb;
    if ((e = armas_x_cholfactor(&A0, &W, &P, flags, &conf)) < 0) {
        printf("..%s.factoring error %d [%d]\n", blk, conf.error, e);
    }

    if (flags & ARMAS_LOWER) {
        flags1 = ARMAS_LOWER|ARMAS_RIGHT;
        flags2 = ARMAS_TRANS|ARMAS_LOWER|ARMAS_RIGHT;
    } else {
        flags2 = ARMAS_UPPER|ARMAS_RIGHT;
        flags1 = ARMAS_UPPER|ARMAS_TRANS|ARMAS_RIGHT;
    }

    // C = I*L*L.T || I*U.T*U
    armas_x_mult_trm(&C, &A0, 1.0, flags1, &conf);
    armas_x_mult_trm(&C, &A0, 1.0, flags2, &conf);
    armas_x_make_trm(&C, flags);
    if (N < 10 && verbose > 1) {
        printf("(1) LL.T or U.TU:\n"); armas_x_printf(stdout, "%6.3f", &C);
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P);
    }
    armas_x_pivot(&C, &P, pflgs|ARMAS_PIVOT_BACKWARD, &conf);
    if (N < 10 && verbose > 1) {
        printf("(2) pivoted: \n"); armas_x_printf(stdout, "%6.3f", &C);
    }
    n0 = rel_error(&n1, &C, &A1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.%s = A\n", PASS(ok), blk, fact);
    if (verbose > 0)
      printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&C);
    armas_x_release(&W);
    armas_pivot_release(&P);
    return ok;
}

int test_cholpv_solve(int M, int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, B, B0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P0;
    DTYPE n0, n1;
    int e, ok;
    char *fact = flags & ARMAS_LOWER ? "LL^T" : "U^TU";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&B0, N, M);
    armas_x_init(&B, N, M);
    armas_x_init(&W, N, 1);
    armas_pivot_init(&P0, N);
    
    armas_x_set_values(&A0, unitrand, 0);
    armas_x_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
    armas_x_mcopy(&A0, &A1);
    armas_x_make_trm(&A0, flags);

    armas_x_set_values(&B0, zeromean, 0);
    // B = A*B0
    armas_x_mult(&B, &A1, &B0, 1.0, 0.0, 0, &conf);

    conf.lb = lb;
    if ((e = armas_x_cholfactor(&A0, &W, &P0, flags, &conf)) < 0) 
        printf("Error: factoring error %d, [%d]\n", conf.error, e);

    if ((e = armas_x_cholsolve(&B, &A0, &P0, flags, &conf)) < 0)
        printf("Error: solver error %d, [%d]\n", conf.error, e);
    if (N < 10) {
        printf("(%s)^-1*B:\n", fact); armas_x_printf(stdout, "%6.3f", &B);
    }
    
    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.(%s)^-1*B = X\n", PASS(ok), blk, fact);
    if (verbose > 0)
      printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&B);
    armas_x_release(&B0);
    armas_x_release(&W);
    armas_pivot_release(&P0);
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

  if (! test_cholpv(N, 0, verbose, ARMAS_LOWER))
    fails++;
  if (! test_cholpv(N, LB, verbose, ARMAS_LOWER))
    fails++;
  if (! test_cholpv(N, 0, verbose, ARMAS_UPPER))
    fails++;
  if (! test_cholpv(N, LB, verbose, ARMAS_UPPER))
    fails++;
  if (! test_cholpv_solve(M, N, 0, verbose, ARMAS_LOWER))
    fails++;
  if (! test_cholpv_solve(M, N, LB, verbose, ARMAS_LOWER))
    fails++;
  if (! test_cholpv_solve(M, N, 0, verbose, ARMAS_UPPER))
    fails++;
  if (! test_cholpv_solve(M, N, LB, verbose, ARMAS_UPPER))
    fails++;
  
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
