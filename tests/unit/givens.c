
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "givens"

// test: QR factorization with Givens rotations
int test_qr(int M, int N, int verbose)
{
  armas_d_dense_t A0, A1, Qt, d;
  int i, j, ok, n, m;
  double c, s, r, nrm, y0, y1;
  armas_conf_t *conf = armas_conf_default();

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&Qt, M, M);

  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);
  armas_d_diag(&d, &Qt, 0);
  armas_d_madd(&d, 1.0, 0);
  
  // R = G(n)...G(2)G(1)*A; Q = G(1)*G(2)...G(n) ;  Q.T = G(n)...G(2)G(1)
  for (j = 0; j < N; j++) {
    // .. zero elements below diagonal, starting from bottom
    for (i = M-2; i >= j; i--) {
      armas_d_gvcompute(&c, &s, &r, armas_d_get(&A0, i, j), armas_d_get(&A0, i+1, j));
      armas_d_set(&A0, i,   j, r);
      armas_d_set(&A0, i+1, j, 0.0);
      
      // apply rotations on this row 
      armas_d_gvleft(&A0, c, s, i, i+1, j+1, N-j-1);
      // update Qt = G(k)*Qt 
      armas_d_gvleft(&Qt, c, s, i, i+1, 0, M);
    }
  }

  // compute A1 = A1 - Q*R
  armas_d_mult(&A1, &Qt, &A0, -1.0, 1.0, ARMAS_TRANSA, conf);

  nrm = armas_d_mnorm(&A1, ARMAS_NORM_ONE, conf);
  ok = isOK(nrm, N);
  printf("%s:  A == Q*R\n", PASS(ok));
  printf("  M=%d, N=%d ||A - Q*R||_1: %e [%d]\n", M, N, nrm, (int)(nrm/DBL_EPSILON));
  return ok;
}


// test: LQ factorization with Givens rotations
int test_lq(int M, int N, int verbose)
{
  armas_d_dense_t A0, A1, Qt, d;
  int i, j, ok, n, m;
  double c, s, r, nrm, y0, y1;
  armas_conf_t *conf = armas_conf_default();

  armas_d_init(&A0, M, N);
  armas_d_init(&A1, M, N);
  armas_d_init(&Qt, N, N);

  armas_d_set_values(&A0, unitrand, ARMAS_ANY);
  armas_d_mcopy(&A1, &A0);
  armas_d_diag(&d, &Qt, 0);
  armas_d_madd(&d, 1.0, 0);
  
  // R = G(n)...G(2)G(1)*A; Q = G(1)*G(2)...G(n) ;  Q.T = G(n)...G(2)G(1)
  n = 0; m = 0;
  for (i = 0; i < M; i++) {
    // .. zero elements right of diagonal, starting from rightmost
    for (j = N-2; j >= i; j--) {
      armas_d_gvcompute(&c, &s, &r, armas_d_get(&A0, i, j), armas_d_get(&A0, i, j+1));
      armas_d_set(&A0, i, j,   r);
      armas_d_set(&A0, i, j+1, 0.0);
      
      // apply rotations on columns j, j+1
      armas_d_gvright(&A0, c, s, j, j+1, i+1, M-i-1);
      // update Qt = G(k)*Qt 
      armas_d_gvright(&Qt, c, s, j, j+1, 0, N);
    }
  }

  // compute A1 = A1 - L*Q
  armas_d_mult(&A1, &A0, &Qt, -1.0, 1.0, ARMAS_TRANSB, conf);

  nrm = armas_d_mnorm(&A1, ARMAS_NORM_ONE, conf);
  ok = isOK(nrm, N);
  printf("%s:  A == L*Q\n", PASS(ok));
  printf("  M=%d, N=%d ||A - L*Q||_1: %e [%d]\n", M, N, nrm, (int)(nrm/DBL_EPSILON));
  return ok;
}

main(int argc, char **argv)
{
  int opt;
  int M = 199;
  int N = 181;
  int K = N;
  int LB = 0;
  int ok = 0;
  int nproc = 1;
  int verbose = 0;

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
  
  if (!test_qr(M, N, verbose))
    fails++;
  if (!test_lq(N, M, verbose))
    fails++;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
