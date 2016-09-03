
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "givens"

// test: QR factorization with Givens rotations
int test_qr(int M, int N, int verbose)
{
  armas_x_dense_t A0, A1, Qt, d;
  int i, j, ok;
  DTYPE c, s, r, nrm, nrm_A;
  armas_conf_t *conf = armas_conf_default();

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&Qt, M, M);

  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);
  armas_x_diag(&d, &Qt, 0);
  armas_x_madd(&d, 1.0, 0);
  nrm_A = armas_x_mnorm(&A1, ARMAS_NORM_ONE, conf);

  // R = G(n)...G(2)G(1)*A; Q = G(1)*G(2)...G(n) ;  Q.T = G(n)...G(2)G(1)
  for (j = 0; j < N; j++) {
    // .. zero elements below diagonal, starting from bottom
    for (i = M-2; i >= j; i--) {
      armas_x_gvcompute(&c, &s, &r, armas_x_get(&A0, i, j), armas_x_get(&A0, i+1, j));
      armas_x_set(&A0, i,   j, r);
      armas_x_set(&A0, i+1, j, 0.0);
      
      // apply rotations on this row 
      armas_x_gvleft(&A0, c, s, i, i+1, j+1, N-j-1);
      // update Qt = G(k)*Qt 
      armas_x_gvleft(&Qt, c, s, i, i+1, 0, M);
    }
  }

  // compute A1 = A1 - Q*R
  armas_x_mult(&A1, &Qt, &A0, -1.0, 1.0, ARMAS_TRANSA, conf);

  nrm = armas_x_mnorm(&A1, ARMAS_NORM_ONE, conf);
  nrm /= nrm_A;

  ok = isOK(nrm, N);
  printf("%s:  A == Q*R\n", PASS(ok));
  printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));
  return ok;
}


// test: LQ factorization with Givens rotations
int test_lq(int M, int N, int verbose)
{
  armas_x_dense_t A0, A1, Qt, d;
  int i, j, ok;
  DTYPE c, s, r, nrm, nrm_A;
  armas_conf_t *conf = armas_conf_default();

  armas_x_init(&A0, M, N);
  armas_x_init(&A1, M, N);
  armas_x_init(&Qt, N, N);

  armas_x_set_values(&A0, unitrand, ARMAS_ANY);
  armas_x_mcopy(&A1, &A0);
  armas_x_diag(&d, &Qt, 0);
  armas_x_madd(&d, 1.0, 0);
  nrm_A = armas_x_mnorm(&A1, ARMAS_NORM_ONE, conf);
  
  // R = G(n)...G(2)G(1)*A; Q = G(1)*G(2)...G(n) ;  Q.T = G(n)...G(2)G(1)
  for (i = 0; i < M; i++) {
    // .. zero elements right of diagonal, starting from rightmost
    for (j = N-2; j >= i; j--) {
      armas_x_gvcompute(&c, &s, &r, armas_x_get(&A0, i, j), armas_x_get(&A0, i, j+1));
      armas_x_set(&A0, i, j,   r);
      armas_x_set(&A0, i, j+1, 0.0);
      
      // apply rotations on columns j, j+1
      armas_x_gvright(&A0, c, s, j, j+1, i+1, M-i-1);
      // update Qt = G(k)*Qt 
      armas_x_gvright(&Qt, c, s, j, j+1, 0, N);
    }
  }

  // compute A1 = A1 - L*Q
  armas_x_mult(&A1, &A0, &Qt, -1.0, 1.0, ARMAS_TRANSB, conf);

  nrm = armas_x_mnorm(&A1, ARMAS_NORM_ONE, conf);
  nrm /= nrm_A;
  ok = isOK(nrm, N);
  printf("%s:  A == L*Q\n", PASS(ok));
  printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));
  return ok;
}

int main(int argc, char **argv)
{
  int opt;
  int M = 199;
  int N = 181;
  int verbose = 0;

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
  } else if (optind < argc-1) {
    N = atoi(argv[optind]);
    M = N;
  } else if (optind < argc) {
    N = atoi(argv[optind]);
    M = N;
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
