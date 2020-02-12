
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"

/*
 * A = [M,K], B = [K,N] --> C = [M,N]
 *
 * test 1:
 *   a. compute C0  = A*B      : [M,K]*[K,N] = [M,N]
 *   b. compute C1  = B.T*A.T  : [N,K]*[K,M] = [N,M]
 *   c. verify  C0 == C1.T
 *
 * test 2:
 *   a. compute C0 = A*B.T     : [M,K]*[N,K]  = [M,N] if K == N
 *   b. compute C1 = B*A.T     : [K,N]*[K,M]  = [N,M] if K == N
 *   c. verify  C0 == C1.T
 *
 * test 3:
 *   a. compute C0 = A.T*B     : [K,M]*[K,N] = [M,N] if K == M
 *   b. compute C1 = B.T*A     : [N,K]*[M,K] = [N,M] if K == M
 *
 */
int main(int argc, char **argv) {

  armas_conf_t conf;
  armas_x_dense_t C, Ct, T, A, B, Ce;

  int ok, opt;
  int N = 33;
  int M = 33;
  int K = 33;
  int fails = 0;
  int normal_prec = 0;
  int debug = 0;
  int prec = 200;
  int verbose = 0;
  double cwant = 1.0/_EPS; // wanted condition number
  double dot, cond, m_one_t, m_one; 

  while ((opt = getopt(argc, argv, "C:p:vSD")) != -1) {
    switch (opt) {
    case 'C':
      cwant = strtod(optarg, (char **)0);
      break;
    case 'D':
      debug = 1;
      break;
    case 'p':
      prec = atoi(optarg);
      break;
    case 'S':
      normal_prec = 1;
      break;
    case 'v':
      verbose += 1;
      break;
    default:
      fprintf(stderr, "usage: ext_gemm [-p bits -C cond -v -S] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc) {
    N = atoi(argv[optind]);
    M = K = N;
  }
  
  conf = *armas_conf_default();

  armas_x_init(&C, M, N);
  armas_x_init(&Ct, N, M);
  armas_x_init(&Ce, M, N);
  armas_x_init(&T, M, N);
  armas_x_set_values(&C, zero, ARMAS_NULL);
  armas_x_set_values(&Ct, zero, ARMAS_NULL);

  // test 1: M != N != K
  armas_x_init(&A, M, K);
  armas_x_init(&B, K, N);
  
  if (debug) {
    printf("conf .mb, .nb, .kb: %d, %d, %d\n", conf.mb, conf.nb, conf.kb);
    armas_x_set_values(&A, one, ARMAS_NULL);
    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mult(0.0, &Ce, 1.0, &A, &B, 0, &conf);
  } else {
    ep_genmat(&dot, &cond, &A, &B, cwant);
    ep_gemm(&Ce, &A, &B, 1.0, 0.0, prec, ARMAS_NULL);
    if (verbose > 0) {
      printf("dot = %e, cond = %e, cwant = %e\n", dot, cond, cwant);
    }
  }

  // extended precision computations
  conf.optflags = ARMAS_OEXTPREC;

  // C = A*B; C.T = B.T*A.T
  armas_x_mult(0.0, &C, 1.0, &A, &B, 0, &conf);
  armas_x_mult(0.0, &Ct, 1.0, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, &conf);

  if (verbose > 2 && N < 10) {
    printf("C.exact:\n"); armas_x_printf(stdout, "%13e", &Ce);
    printf("A*B:\n"); armas_x_printf(stdout, "%13e", &C);
    printf("(B.T*A.T).T:\n"); armas_x_printf(stdout, "%13e", &Ct);
  }

  armas_x_transpose(&T, &Ct);
  m_one   = rel_error((DTYPE *)0, &C, &Ce, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  m_one_t = rel_error((DTYPE *)0, &T, &Ce, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  if (verbose > 1 && N < 10) {
    printf("Ce - A*B:\n"); armas_x_printf(stdout, "%13e", &C);
    printf("Ce - (B.T*A.T).T:\n"); armas_x_printf(stdout, "%13e", &T);
  }

  ok = m_one < N*_EPS2; 
  printf("%-6s: ||C.exact - A*B||/||C.exact|| < N*eps^2\n",  ok ? "OK" : "FAILED");
  fails += 1 - ok;

  ok = m_one_t < N*_EPS2;
  printf("%-6s: ||C.exact - (B.T*A.T).T||/||C.exact|| < N*eps^2\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  if (verbose > 0) {
    printf("relative error, extendend precision\n");
    printf("  ||C - A*B||_1/||C||_1        : %e\n", m_one);
    printf("  ||C - (B.T*A.T).T||_1/||C||_1: %e\n", m_one_t);
  }

  if (normal_prec && verbose > 0) {
    // normal precision computations; these will fail mostly
    conf.optflags ^= ARMAS_OEXTPREC;

    // C = A*B; C.T = B.T*A.T
    armas_x_mult(0.0, &C, 1.0, &A, &B, 0, &conf);
    armas_x_mult(0.0, &Ct,1.0, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, &conf);

    armas_x_transpose(&T, &Ct);
    m_one   = rel_error((DTYPE *)0, &C, &Ce, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    m_one_t = rel_error((DTYPE *)0, &T, &Ce, ARMAS_NORM_ONE, ARMAS_NONE, &conf);

    ok = m_one < N*_EPS;
    printf("\ncompare to normal precision\n");
    printf("%-6s: ||C.exact - A*B||/||C.exact|| < N*eps\n", ok ? "OK" : "FAILED");
    ok = m_one_t < N*_EPS;
    printf("%-6s: ||C.exact - (B.T*A.T).T||/||C.exact|| < N*eps\n",  ok ? "OK" : "FAILED");
    
    printf("relative error, normal precision\n");
    printf("  ||C - A*B||_1/||C||_1        : %e\n", m_one);
    printf("  ||C - (B.T*A.T).T||_1/||C||_1: %e\n", m_one_t);
  }

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
