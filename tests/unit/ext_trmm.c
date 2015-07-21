
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

extern void
ep_gentrm(double *dot, double *tcond, armas_d_dense_t *A, armas_d_dense_t *B, double cond, int flags);

// ne = norm1 of exact; re = norm1 of exact-result
void compute(double *ne, double *re,
             armas_d_dense_t *B, armas_d_dense_t *A, armas_d_dense_t *C,
             int flags, int prec, int verbose, armas_conf_t *conf)
{
  int bits = 0;

  switch (flags & (ARMAS_RIGHT|ARMAS_TRANS)) {
  case ARMAS_RIGHT|ARMAS_TRANS:
    bits = ARMAS_TRANSB;
    break;
  case ARMAS_TRANS:
    bits = ARMAS_TRANSA;
    break;
  case ARMAS_RIGHT:
    break;
  default:
    break;
  }

  if (flags & ARMAS_RIGHT) {
    ep_gemm(C, B, A, 1.0, 0.0, prec, bits);
  } else {
    ep_gemm(C, A, B, 1.0, 0.0, prec, bits);
  }
  *ne = armas_d_mnorm(C, ARMAS_NORM_ONE, conf);

  // B = A*B
  armas_d_mult_trm(B, A, 1.0, flags, conf);
  if (verbose > 1 && A->rows < 10) {
    printf("exc(A*B):\n"); armas_d_printf(stdout, "%13e", C);
    printf("trm(A)*B:\n"); armas_d_printf(stdout, "%13e", B);
  }
  armas_d_scale_plus(B, C, 1.0, -1.0, ARMAS_NONE, conf);

  *re = armas_d_mnorm(B, ARMAS_NORM_ONE, conf);
}

int test(char *name, int N, int K, int flags, int verbose, int prec, double cwant, armas_conf_t *conf)
{
  armas_d_dense_t C, Ct, T, B0, A, B, Ce;
  double dot, cond, m_c, m_one;
  int ok;

  armas_d_init(&Ce, N, K);

  armas_d_init(&A, N, N);
  armas_d_init(&B, N, K);
  armas_d_init(&B0, N, K);
  
  ep_gentrm(&dot, &cond, &A, &B, cwant, flags);
  armas_d_make_trm(&A, flags);
  armas_d_mcopy(&B0, &B);
  if (verbose > 1 && N < 10) {
    printf("A:\n"); armas_d_printf(stdout, "%13e", &A);
    if (K < 10)
      printf("B:\n"); armas_d_printf(stdout, "%13e", &B);
  }

  // extended precision computations

  // 1. A*B
  compute(&m_c, &m_one, &B, &A, &Ce, flags, prec, verbose, conf);

  ok = m_one < N*DBL_EPSILON;
  printf("%-4s: %s rel.error %e [%e/%e]\n",  PASS(ok), name, m_one/m_c, m_one, m_c);
  if (!ok && N < 10) {
    printf("B-Ce:\n"); armas_d_printf(stdout, "%13e", &B);
  }

  armas_d_release(&Ce);
  armas_d_release(&A);
  armas_d_release(&B);
  armas_d_release(&B0);

  return ok;
}

/*
 *
 */
main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, Ct, T, B0, A, B, Ce;

  int ok, opt;
  int N = 33;
  int M = 33;
  int K = 33;
  int fails = 0;
  int normal_prec = 0;
  int e0, e1;
  int prec = 200;
  int verbose = 0;
  int flags = ARMAS_UPPER;
  int all = 0;
  double cwant = 1e14; // wanted condition number
  double dot, cond, m_one_t, m_one, m_c;

  while ((opt = getopt(argc, argv, "C:p:vASLRT")) != -1) {
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
    case 'A':
      all = 1;
      break;
    case 'L':
      flags &= ~ARMAS_UPPER;
      flags |= ARMAS_LOWER;
      break;
    case 'R':
      flags &= ~ARMAS_LEFT;
      flags |= ARMAS_RIGHT;
      break;
    case 'T':
      flags |= ARMAS_TRANS;
      break;
    default:
      fprintf(stderr, "usage: ext_gemm [-p bits -C cond -v -SLRT] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc-1) {
    N = atoi(argv[optind]);
    K = atoi(argv[optind+1]);
  } else if (optind < argc) {
    N = atoi(argv[optind]);
    M = K = N;
  }
  
  if (verbose > 1) {
    printf("flags = 0x%X\n", flags);
  }
  conf = *armas_conf_default();
  conf.optflags = ARMAS_OEXTPREC;

  if (!all) {
    ok = test("single", N, K, flags, verbose, prec, cwant, &conf);
    fails += 1 - ok;
  } else {
    ok = test("u(A)*B  ", N, K, ARMAS_UPPER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("l(A)*B  ", N, K, ARMAS_LOWER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("u(A.T)*B", N, K, ARMAS_UPPER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("l(A.T)*B", N, K, ARMAS_LOWER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*u(A)  ", N, K, ARMAS_RIGHT|ARMAS_UPPER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*l(A)  ", N, K, ARMAS_RIGHT|ARMAS_LOWER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*u(A.T)", N, K, ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*l(A.T)", N, K, ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
  }


#if 0
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
#endif

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
