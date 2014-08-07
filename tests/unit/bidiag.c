
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "bidiag"

int test_reduce(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, tauq0, taup0, tauq1, taup1, W, tmp;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize;
  double nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&tauq0, imin(M, N), 1);
  armas_d_init(&tauq1, imin(M, N), 1);
  armas_d_init(&taup0, imin(M, N), 1);
  armas_d_init(&taup1, imin(M, N), 1);

  conf.lb = lb;
  wsize = armas_d_bdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  // unblocked reduction
  conf.lb = 0;
  armas_d_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // blocked reduction
  conf.lb = lb;
  armas_d_bdreduce(&A1, &tauq1, &taup1, &W, &conf);

  armas_d_scale_plus(&A0, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);

  ok = isFINE(nrm, N*1e-12);
  printf("%s: %s unblk.BD(A) == blk.BD(A)\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  ||unblk.BD(A) - blk.BD(A)||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
    armas_d_axpy(&tauq0, &tauq1, -1.0, &conf);
    nrm = armas_d_nrm2(&tauq0, &conf);
    printf("  ||unblk.BD.tauq - blk.BD.tauq||: %e\n", nrm);
    armas_d_axpy(&taup0, &taup1, -1.0, &conf);
    nrm = armas_d_nrm2(&taup0, &conf);
    printf("  ||unblk.BD.taup - blk.BD.taup||: %e\n", nrm);
  }

  armas_d_release(&A0);
  armas_d_release(&A1);
  armas_d_release(&tauq0);
  armas_d_release(&tauq1);
  armas_d_release(&taup0);
  armas_d_release(&taup1);
  armas_d_release(&W);

  return ok;
}

// compute: ||A - Q*B*P.T|| == O(eps)
int test_mult_qpt(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, B, tauq0, taup0, W, tmp, Btmp, d0, d1;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize, err, ws2;
  double nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&B, M, N);
  armas_d_init(&tauq0, M, 1);
  armas_d_init(&taup0, N, 1);

  conf.lb = lb;
  wsize = armas_d_bdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  armas_d_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // extract B from A
  armas_d_mcopy(&B, &A0);
  if (M > N) {
    // zero subdiagonal entries
    armas_d_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_d_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above 1st superdiagonal
    armas_d_submatrix(&Btmp, &B, 0, 1, N-1, N-1);
    armas_d_make_trm(&Btmp, ARMAS_LOWER);
  } else {
    // zero entries below 1st subdiagonal
    armas_d_submatrix(&Btmp, &B, 1, 0, M-1, M-1);
    armas_d_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above diagonal
    armas_d_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_d_make_trm(&Btmp, ARMAS_LOWER);
  }

  // A = Q*B*P.T; 
  armas_d_bdmult(&B, &A0, &tauq0, &W, ARMAS_LEFT|ARMAS_MULTQ, &conf);
  armas_d_bdmult(&B, &A0, &taup0, &W, ARMAS_RIGHT|ARMAS_TRANS|ARMAS_MULTP, &conf);

  // B == A1?
  armas_d_scale_plus(&B, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);

  ok = isFINE(nrm, N*1e-12);
  printf("%s: %s  Q*B*P.T == A\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  ||A - Q*B*P.T||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&B);
  armas_d_release(&tauq0);
  armas_d_release(&taup0);
  armas_d_release(&W);
  return ok;
}

// compute: ||B - Q.T*A*P|| == O(eps)
int test_mult_qtp(int M, int N, int lb, int verbose)
{
  armas_d_dense_t A0, A1, B, tauq0, taup0, W, tmp, Btmp, d0, d1;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize, err, ws2;
  double nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&B, M, N);
  armas_d_init(&tauq0, M, 1);
  armas_d_init(&taup0, N, 1);

  conf.lb = lb;
  wsize = armas_d_bdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  armas_d_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  // extract B from A
  armas_d_mcopy(&B, &A0);
  if (M > N) {
    // zero subdiagonal entries
    armas_d_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_d_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above 1st superdiagonal
    armas_d_submatrix(&Btmp, &B, 0, 1, N-1, N-1);
    armas_d_make_trm(&Btmp, ARMAS_LOWER);
  } else {
    // zero entries below 1st subdiagonal
    armas_d_submatrix(&Btmp, &B, 1, 0, M-1, M-1);
    armas_d_make_trm(&Btmp, ARMAS_UPPER);
    // zero entries above diagonal
    armas_d_submatrix(&Btmp, &B, 0, 0, M, N);
    armas_d_make_trm(&Btmp, ARMAS_LOWER);
  }

  // B = Q.T*B*P; 
  armas_d_bdmult(&A1, &A0, &tauq0, &W, ARMAS_LEFT|ARMAS_MULTQ|ARMAS_TRANS, &conf);
  armas_d_bdmult(&A1, &A0, &taup0, &W, ARMAS_RIGHT|ARMAS_MULTP, &conf);

  // B == A1?
  armas_d_scale_plus(&B, &A1, 1.0, -1.0, ARMAS_NONE, &conf);
  nrm = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);

  ok = isFINE(nrm, N*1e-12);
  printf("%s: %s  B == Q.T*A*P\n", PASS(ok), mbyn);
  if (verbose > 0) {
    printf("  ||B - Q.T*B*P||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
  }

  armas_d_release(&A0);
  armas_d_release(&B);
  armas_d_release(&tauq0);
  armas_d_release(&taup0);
  armas_d_release(&W);
  return ok;
}

// compute: ||B - Q.T*A*P|| == O(eps)
int test_build_qp(int M, int N, int lb, int K, int flags, int verbose)
{
  armas_d_dense_t A0, tauq0, taup0, W, tmp, Qh, QQt, d0, d1;
  armas_conf_t conf = *armas_conf_default();
  int ok, wsize, err, ws2;
  double nrm;
  char *mbyn = M >= N ? "M >= N" : "M < N";

  armas_d_init(&A0, M, N);
  armas_d_init(&tauq0, imin(M, N), 1);
  armas_d_init(&taup0, imin(M, N), 1);

  conf.lb = lb;
  wsize = armas_d_bdreduce_work(&A0, &conf);
  armas_d_init(&W, wsize, 1);

  // set source data
  armas_d_set_values(&A0, unitrand, ARMAS_ANY);

  // reduce to bidiagonal matrix
  conf.lb = lb;
  armas_d_bdreduce(&A0, &tauq0, &taup0, &W, &conf);

  conf.error = 0;
  if (flags & ARMAS_WANTQ) {
    armas_d_bdbuild(&A0, &tauq0, &W, K, flags, &conf);

    if (M < N) {
      armas_d_init(&QQt, M, M);
      armas_d_submatrix(&Qh, &A0, 0, 0, M, M);
      armas_d_mult(&QQt, &Qh, &Qh, 1.0, 0.0, ARMAS_TRANSA, &conf);
    } else {
      armas_d_init(&QQt, N, N);
      armas_d_mult(&QQt, &A0, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    }
    armas_d_diag(&d0, &QQt, 0);
    armas_d_madd(&d0, -1.0, ARMAS_NONE);

    nrm = armas_d_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
    ok = isFINE(nrm, N*1e-12);
    printf("%s: %s  I == Q.T*Q\n", PASS(ok), mbyn);
    if (verbose > 0) {
      printf("  ||I - Q.T*Q||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
    }
  } else {
    // P matrix
    armas_d_bdbuild(&A0, &taup0, &W, K, flags, &conf);

    if (N < M) {
      armas_d_init(&QQt, N, N);
      armas_d_submatrix(&Qh, &A0, 0, 0, N, N);
      armas_d_mult(&QQt, &Qh, &Qh, 1.0, 0.0, ARMAS_TRANSA, &conf);
    } else {
      armas_d_init(&QQt, M, M);
      armas_d_mult(&QQt, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
    }
    armas_d_diag(&d0, &QQt, 0);
    armas_d_madd(&d0, -1.0, ARMAS_NONE);

    nrm = armas_d_mnorm(&QQt, ARMAS_NORM_ONE, &conf);
    
    ok = isFINE(nrm, N*1e-12);
    printf("%s: %s  I == P*P.T\n", PASS(ok), mbyn);
    if (verbose > 0) {
      printf("  ||I - P*P.T||: %e [%d]\n", nrm, (int)(nrm/DBL_EPSILON));
    }
  }

  armas_d_release(&A0);
  armas_d_release(&tauq0);
  armas_d_release(&taup0);
  armas_d_release(&W);
  armas_d_release(&QQt);
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 787;
  int N = 741;
  int K = N;
  int LB = 36;
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
