
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"
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
main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, Ct, T, C0, A, B, Ce;

  int ok, opt;
  int N = 33;
  int M = 33;
  int K = 33;
  int fails = 0;
  int normal_prec = 0;
  int e0, e1;
  int prec = 200;
  int verbose = 0;
  double cwant = 1e14; // wanted condition number
  double dot, cond, m_one_t, m_one, m_c;

  while ((opt = getopt(argc, argv, "C:p:vS")) != -1) {
    switch (opt) {
    case 'C':
      cwant = strtod(optarg, (char **)0);
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
  
  conf.optflags = ARMAS_OEXTPREC;

  armas_d_init(&C, M, N);
  armas_d_init(&Ct, N, M);
  armas_d_init(&Ce, M, N);
  armas_d_init(&T, M, N);
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_set_values(&Ct, zero, ARMAS_NULL);

  // test 1: M != N != K
  armas_d_init(&A, M, K);
  armas_d_init(&B, K, N);
  
  ep_genmat(&dot, &cond, &A, &B, cwant);
  ep_gemm(&Ce, &A, &B, 1.0, 0.0, prec, ARMAS_NULL);

  m_c = armas_d_mnorm(&Ce, ARMAS_NORM_ONE, &conf);

  // extended precision computations

  // C = A*B; C.T = B.T*A.T
  armas_d_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
  armas_d_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);

  armas_d_transpose(&T, &Ct);
  armas_d_scale_plus(&T, &Ce, 1.0, -1.0, ARMAS_NONE, &conf);
  armas_d_scale_plus(&C, &Ce, 1.0, -1.0, ARMAS_NONE, &conf);

  if (verbose > 1 && N < 10) {
    printf("Ce - A*B:\n"); armas_d_printf(stdout, "%13e", &C);
    printf("Ce - (B.T*A.T).T:\n"); armas_d_printf(stdout, "%13e", &T);
  }
  m_one_t = armas_d_mnorm(&T, ARMAS_NORM_ONE, &conf);
  m_one   = armas_d_mnorm(&C, ARMAS_NORM_ONE, &conf);

  ok = m_one < N*DBL_EPSILON && m_one_t < N*DBL_EPSILON;
  printf("%-6s: ||C.exact - A*B|| ~ 0 && ||C.exact - (B.T*A.T).T|| ~ 0\n",
         ok ? "OK" : "FAILED");

  if (verbose > 0) {
    printf("relative error, extendend precision\n");
    printf("  ||C - A*B||_1/||C||_1        : %e [%e/%e]\n", m_one/m_c, m_one, m_c);
    printf("  ||C - (B.T*A.T).T||_1/||C||_1: %e [%e/%e]\n", m_one_t/m_c,m_one_t, m_c);
  }
  fails += 1 - ok;

  if (normal_prec && verbose > 0) {
    // normal precision computations; these will fail mostly
    conf.optflags ^= ARMAS_OEXTPREC;

    // C = A*B; C.T = B.T*A.T
    armas_d_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
    armas_d_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);

    armas_d_transpose(&T, &Ct);
    armas_d_scale_plus(&T, &Ce, 1.0, -1.0, ARMAS_NONE, &conf);
    armas_d_scale_plus(&C, &Ce, 1.0, -1.0, ARMAS_NONE, &conf);

    m_one_t = armas_d_mnorm(&T, ARMAS_NORM_ONE, &conf);
    m_one   = armas_d_mnorm(&C, ARMAS_NORM_ONE, &conf);

    ok = m_one < 1000*N*DBL_EPSILON && m_one_t < 1000*N*DBL_EPSILON;
    printf("\ncompare to normal precision\n");
    printf("%-6s: ||C.exact - A*B|| ~ 0 && ||C.exact - (B.T*A.T).T|| ~ 0\n",
           ok ? "OK" : "FAILED");
    
    printf("relative error, normal precision\n");
    printf("  ||C - A*B||_1/||C||_1        : %e [%e/%e]\n", m_one/m_c, m_one, m_c);
    printf("  ||C - (B.T*A.T).T||_1/||C||_1: %e [%e/%e]\n", m_one_t/m_c,m_one_t, m_c);
  }

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
