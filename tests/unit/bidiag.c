
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
  __Matrix A0, A1, tauq0, taup0, tauq1, taup1, W;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&tauq0, imin(M, N), 1);
  matrix_init(&tauq1, imin(M, N), 1);
  matrix_init(&taup0, imin(M, N), 1);
  matrix_init(&taup1, imin(M, N), 1);

  conf.lb = lb;
  wsize = matrix_bdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  // unblocked reduction
  conf.lb = 0;
  matrix_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // blocked reduction
  conf.lb = lb;
  matrix_bdreduce(&A1, &tauq1, &taup1, &W, &conf);

  nrm = rel_error((__Dtype *)0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: %s unblk.BD(A) == blk.BD(A)\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  ||  error.BD(A)  ||: %e [%d]\n", nrm, ndigits(nrm));
    nrm = rel_error((__Dtype *)0, &tauq0, &tauq1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    printf("  || error.BD.tauq ||: %e [%d]\n", nrm, ndigits(nrm));
    nrm = rel_error((__Dtype *)0, &taup0, &taup1, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    printf("  || error.BD.taup ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&A1);
  matrix_release(&tauq0);
  matrix_release(&tauq1);
  matrix_release(&taup0);
  matrix_release(&taup1);
  matrix_release(&W);

  return ok;
}

// compute: ||A - Q*B*P.T|| == O(eps)
int test_mult_qpt(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1, B, tauq0, taup0, W, Btmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&B, M, N);
  matrix_init(&tauq0, M, 1);
  matrix_init(&taup0, N, 1);

  conf.lb = lb;
  wsize = matrix_bdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  matrix_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // extract B from A
  matrix_mcopy(&B, &A0);
  if (M > N) {
    // zero subdiagonal entries
    matrix_submatrix(&Btmp, &B, 0, 0, M, N);
    matrix_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above 1st superdiagonal
    matrix_submatrix(&Btmp, &B, 0, 1, N-1, N-1);
    matrix_make_trm(&Btmp, ARMAS_LOWER);
  } else {
    // zero entries below 1st subdiagonal
    matrix_submatrix(&Btmp, &B, 1, 0, M-1, M-1);
    matrix_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above diagonal
    matrix_submatrix(&Btmp, &B, 0, 0, M, N);
    matrix_make_trm(&Btmp, ARMAS_LOWER);
  }

  // A = Q*B*P.T; 
  matrix_bdmult(&B, &A0, &tauq0, &W, ARMAS_LEFT|ARMAS_MULTQ, &conf);
  matrix_bdmult(&B, &A0, &taup0, &W, ARMAS_RIGHT|ARMAS_TRANS|ARMAS_MULTP, &conf);

  nrm = rel_error((__Dtype *)0, &B, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: %s  Q*B*P.T == A\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&B);
  matrix_release(&tauq0);
  matrix_release(&taup0);
  matrix_release(&W);
  return ok;
}

// compute: ||B - Q.T*A*P|| == O(eps)
int test_mult_qtp(int M, int N, int lb, int verbose)
{
  __Matrix A0, A1, B, tauq0, taup0, W, Btmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  matrix_init(&A0, M, N);
  matrix_init(&A1, M, N);
  matrix_init(&B, M, N);
  matrix_init(&tauq0, M, 1);
  matrix_init(&taup0, N, 1);

  conf.lb = lb;
  wsize = matrix_bdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);
  matrix_mcopy(&A1, &A0);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  matrix_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // extract B from A
  matrix_mcopy(&B, &A0);
  if (M > N) {
    // zero subdiagonal entries
    matrix_submatrix(&Btmp, &B, 0, 0, M, N);
    matrix_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above 1st superdiagonal
    matrix_submatrix(&Btmp, &B, 0, 1, N-1, N-1);
    matrix_make_trm(&Btmp, ARMAS_LOWER);
  } else {
    // zero entries below 1st subdiagonal
    matrix_submatrix(&Btmp, &B, 1, 0, M-1, M-1);
    matrix_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above diagonal
    matrix_submatrix(&Btmp, &B, 0, 0, M, N);
    matrix_make_trm(&Btmp, ARMAS_LOWER);
  }

  // B = Q.T*B*P; 
  matrix_bdmult(&A1, &A0, &tauq0, &W, ARMAS_LEFT|ARMAS_MULTQ|ARMAS_TRANS, &conf);
  matrix_bdmult(&A1, &A0, &taup0, &W, ARMAS_RIGHT|ARMAS_MULTP, &conf);

  nrm = rel_error((__Dtype *)0, &B, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = isFINE(nrm, N*__ERROR);
  printf("%s: %s  B == Q.T*A*P\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
  }

  matrix_release(&A0);
  matrix_release(&B);
  matrix_release(&tauq0);
  matrix_release(&taup0);
  matrix_release(&W);
  return ok;
}

// compute: ||B - Q.T*A*P|| == O(eps)
int test_build_qp(int M, int N, int lb, int K, int flags, int verbose)
{
  __Matrix A0, tauq0, taup0, W, Qh, QQt, d0;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  __Dtype nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  matrix_init(&A0, M, N);
  matrix_init(&tauq0, imin(M, N), 1);
  matrix_init(&taup0, imin(M, N), 1);

  conf.lb = lb;
  wsize = matrix_bdreduce_work(&A0, &conf);
  matrix_init(&W, wsize, 1);

  // set source data
  matrix_set_values(&A0, unitrand, ARMAS_ANY);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  matrix_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  conf.error = 0;
  if (flags & ARMAS_WANTQ) {
    matrix_bdbuild(&A0, &tauq0, &W, K, flags, &conf);

    if (M < N) {
      matrix_init(&QQt, M, M);
      matrix_submatrix(&Qh, &A0, 0, 0, M, M);
      matrix_mult(&QQt, &Qh, &Qh, 1.0, 0.0, ARMAS_TRANSA, &conf);
    } else {
      matrix_init(&QQt, N, N);
      matrix_mult(&QQt, &A0, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    }
    matrix_diag(&d0, &QQt, 0);
    matrix_madd(&d0, -1.0, ARMAS_NONE);

    nrm = matrix_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: %s  I == Q.T*Q\n", PASS(ok), mbyn);
    if (verbose > 0) {
      printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
  } else {
    // P matrix
    matrix_bdbuild(&A0, &taup0, &W, K, flags, &conf);

    if (N < M) {
      matrix_init(&QQt, N, N);
      matrix_submatrix(&Qh, &A0, 0, 0, N, N);
      matrix_mult(&QQt, &Qh, &Qh, 1.0, 0.0, ARMAS_TRANSA, &conf);
    } else {
      matrix_init(&QQt, M, M);
      matrix_mult(&QQt, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
    }
    matrix_diag(&d0, &QQt, 0);
    matrix_madd(&d0, -1.0, ARMAS_NONE);

    nrm = matrix_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: %s  I == P*P.T\n", PASS(ok), mbyn);
    if (verbose > 0) {
      printf("  ||  rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
  }

  matrix_release(&A0);
  matrix_release(&tauq0);
  matrix_release(&taup0);
  matrix_release(&W);
  matrix_release(&QQt);
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
  if (! test_build_qp(N, M, LB, N/2, ARMAS_WANTQ, verbose))
    fails++;
  if (! test_build_qp(N, M, LB, N/2, ARMAS_WANTP, verbose))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
