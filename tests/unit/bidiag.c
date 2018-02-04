
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR 6e-6
#else
#define __ERROR 1e-12
#endif

#define NAME "bidiag"

int test_reduce(int M, int N, int lb, int verbose)
{
  armas_x_dense_t A0, A1, tauq0, taup0, tauq1, taup1, W;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  DTYPE nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&tauq0, imin(M, N), 1);
  armas_x_init(&tauq1, imin(M, N), 1);
  armas_x_init(&taup0, imin(M, N), 1);
  armas_x_init(&taup1, imin(M, N), 1);

  conf.lb = lb;
  wsize = armas_x_bdreduce_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);

  // unblocked reduction
  conf.lb = 0;
  armas_x_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // blocked reduction
  conf.lb = lb;
  armas_x_bdreduce(&A1, &tauq1, &taup1, &W, &conf);

  nrm = rel_error((DTYPE *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: %s unblk.BD(A) == blk.BD(A)\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  ||  error.BD(A)  ||: %e [%d]\n", nrm, ndigits(nrm));
    nrm = rel_error((DTYPE *)0, &tauq0, &tauq1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    printf("  || error.BD.tauq ||: %e [%d]\n", nrm, ndigits(nrm));
    nrm = rel_error((DTYPE *)0, &taup0, &taup1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    printf("  || error.BD.taup ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&A1);
  armas_x_release(&tauq0);
  armas_x_release(&tauq1);
  armas_x_release(&taup0);
  armas_x_release(&taup1);
  armas_x_release(&W);

  return ok;
}

// compute: ||A - Q*B*P.T|| == O(eps)
int test_mult_qpt(int M, int N, int lb, int verbose)
{
  armas_x_dense_t A0, A1, B, tauq0, taup0, W, Btmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  DTYPE nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&B, M, N);
  armas_x_init(&tauq0, M, 1);
  armas_x_init(&taup0, N, 1);

  conf.lb = lb;
  wsize = armas_x_bdreduce_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  armas_x_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // extract B from A
  armas_x_mcopy(&B, &A0);
  if (M >= N) {
    // zero subdiagonal entries
    armas_x_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_x_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above 1st superdiagonal
    armas_x_submatrix(&Btmp, &B, 0, 1, N-1, N-1);
    armas_x_make_trm(&Btmp, ARMAS_LOWER);
  } else {
    // zero entries below 1st subdiagonal
    armas_x_submatrix(&Btmp, &B, 1, 0, M-1, M-1);
    armas_x_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above diagonal
    armas_x_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_x_make_trm(&Btmp, ARMAS_LOWER);
  }

  // A = Q*B*P.T; 
  armas_x_bdmult(&B, &A0, &tauq0, &W, ARMAS_LEFT|ARMAS_MULTQ, &conf);
  armas_x_bdmult(&B, &A0, &taup0, &W, ARMAS_RIGHT|ARMAS_TRANS|ARMAS_MULTP, &conf);

  nrm = rel_error((DTYPE *)0, &B, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: %s  Q*B*P.T == A\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&B);
  armas_x_release(&tauq0);
  armas_x_release(&taup0);
  armas_x_release(&W);
  return ok;
}

// compute: ||B - Q.T*A*P|| == O(eps)
int test_mult_qtp(int M, int N, int lb, int verbose)
{
  armas_x_dense_t A0, A1, B, tauq0, taup0, W, Btmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  DTYPE nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&B, M, N);
  armas_x_init(&tauq0, M, 1);
  armas_x_init(&taup0, N, 1);

  conf.lb = lb;
  wsize = armas_x_bdreduce_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  armas_x_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // extract B from A
  armas_x_mcopy(&B, &A0);
  if (M >= N) {
    // zero subdiagonal entries
    armas_x_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_x_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above 1st superdiagonal
    armas_x_submatrix(&Btmp, &B, 0, 1, N-1, N-1);
    armas_x_make_trm(&Btmp, ARMAS_LOWER);
  } else {
    // zero entries below 1st subdiagonal
    armas_x_submatrix(&Btmp, &B, 1, 0, M-1, M-1);
    armas_x_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above diagonal
    armas_x_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_x_make_trm(&Btmp, ARMAS_LOWER);
  }

  // B = Q.T*B*P; 
  armas_x_bdmult(&A1, &A0, &tauq0, &W, ARMAS_LEFT|ARMAS_MULTQ|ARMAS_TRANS, &conf);
  armas_x_bdmult(&A1, &A0, &taup0, &W, ARMAS_RIGHT|ARMAS_MULTP, &conf);

  nrm = rel_error((DTYPE *)0, &B, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: %s  B == Q.T*A*P\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  armas_x_release(&A0);
  armas_x_release(&B);
  armas_x_release(&tauq0);
  armas_x_release(&taup0);
  armas_x_release(&W);
  return ok;
}

// compute: ||B - Q.T*A*P|| == O(eps)
int test_build_qp(int M, int N, int lb, int K, int flags, int verbose)
{
  armas_x_dense_t A0, tauq0, taup0, W, Qh, QQt, d0;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  DTYPE nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_x_init(&A0, M, N);
  armas_x_init(&tauq0, imin(M, N), 1);
  armas_x_init(&taup0, imin(M, N), 1);

  conf.lb = lb;
  wsize = armas_x_bdreduce_work(&A0, &conf);
  armas_x_init(&W, wsize, 1);

  // set source data
  armas_x_set_values(&A0, unitrand, ARMAS_ANY);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  armas_x_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  conf.error = 0;
  if (flags & ARMAS_WANTQ) {
    armas_x_bdbuild(&A0, &tauq0, &W, K, flags, &conf);

    if (M < N) {
      armas_x_init(&QQt, M, M);
      armas_x_submatrix(&Qh, &A0, 0, 0, M, M);
      armas_x_mult(0.0, &QQt, 1.0, &Qh, &Qh, ARMAS_TRANSA, &conf);
    } else {
      armas_x_init(&QQt, N, N);
      armas_x_mult(0.0, &QQt, 1.0, &A0, &A0, ARMAS_TRANSA, &conf);
    }
    armas_x_diag(&d0, &QQt, 0);
    armas_x_madd(&d0, -1.0, ARMAS_NONE);

    nrm = armas_x_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: %s  I == Q.T*Q\n", PASS(ok), mbyn);
    if (verbose > 0) {
      printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
  } else {
    // P matrix
    armas_x_bdbuild(&A0, &taup0, &W, K, flags, &conf);

    if (N < M) {
      armas_x_init(&QQt, N, N);
      armas_x_submatrix(&Qh, &A0, 0, 0, N, N);
      armas_x_mult(0.0, &QQt, 1.0, &Qh, &Qh, ARMAS_TRANSA, &conf);
    } else {
      armas_x_init(&QQt, M, M);
      armas_x_mult(0.0, &QQt, 1.0, &A0, &A0, ARMAS_TRANSB, &conf);
    }
    armas_x_diag(&d0, &QQt, 0);
    armas_x_madd(&d0, -1.0, ARMAS_NONE);

    nrm = armas_x_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: %s  I == P*P.T\n", PASS(ok), mbyn);
    if (verbose > 0) {
      printf("  ||  rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
  }

  armas_x_release(&A0);
  armas_x_release(&tauq0);
  armas_x_release(&taup0);
  armas_x_release(&W);
  armas_x_release(&QQt);
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
  
  if (! test_reduce(M, N, LB, verbose))
    fails++;
  if (! test_reduce(N, M, LB, verbose))
    fails++;
  if (! test_mult_qpt(M, N, LB, verbose))
    fails++;
  if (! test_mult_qpt(N, M, LB, verbose))
    fails++;
  if (! test_mult_qtp(M, N, LB, verbose))
    fails++;
  if (! test_mult_qtp(N, M, LB, verbose))
    fails++;
  if (! test_build_qp(M, N, LB, N/2, ARMAS_WANTQ, verbose))
    fails++;
  if (! test_build_qp(M, N, LB, N/2, ARMAS_WANTP, verbose))
    fails++;
  if (! test_build_qp(N, M, LB, N, ARMAS_WANTQ, verbose))
    fails++;
  if (! test_build_qp(N, M, LB, N, ARMAS_WANTP, verbose))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
