
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"

#if 0
#include <armas/dmatrix.h>
#include "helper.h"

extern void
ep_gentrm(double *dot, double *tcond, armas_d_dense_t *A, armas_d_dense_t *B, double cond, int flags);
#endif

// ne = norm1 of exact; re = |exact-result|/|exact|
void compute(__Dtype *ne, __Dtype *re,
             __Matrix *B, __Matrix *A, __Matrix *C,
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
  //*ne = armas_d_mnorm(C, ARMAS_NORM_ONE, conf);

  // B = A*B
  matrix_mult_trm(B, A, 1.0, flags, conf);
  if (verbose > 1 && A->rows < 10) {
    printf("exc(A*B):\n"); matrix_printf(stdout, "%13e", C);
    printf("trm(A)*B:\n"); matrix_printf(stdout, "%13e", B);
  }
  *re = rel_error(ne, B, C, ARMAS_NORM_ONE, ARMAS_NONE, conf);
#if 0
  armas_d_scale_plus(B, C, 1.0, -1.0, ARMAS_NONE, conf);

  *re = armas_d_mnorm(B, ARMAS_NORM_ONE, conf);
#endif
}

int test(char *name, int N, int K, int flags, int verbose, int prec, double cwant, armas_conf_t *conf)
{
  __Matrix B0, A, B, Ce;
  __Dtype m_c, m_one;
  double dot, cond;
  int ok;

  matrix_init(&Ce, N, K);

  matrix_init(&A, N, N);
  matrix_init(&B, N, K);
  matrix_init(&B0, N, K);
  
  ep_gentrm(&dot, &cond, &A, &B, cwant, flags);
  matrix_make_trm(&A, flags);
  matrix_mcopy(&B0, &B);
  if (verbose > 1 && N < 10) {
    printf("A:\n"); matrix_printf(stdout, "%13e", &A);
    if (K < 10)
      printf("B:\n"); matrix_printf(stdout, "%13e", &B);
  }

  // extended precision computations

  // 1. A*B
  compute(&m_c, &m_one, &B, &A, &Ce, flags, prec, verbose, conf);

  ok = m_one < N*DBL_EPSILON;
  printf("%-4s: %s rel.error %e [%e]\n",  PASS(ok), name, m_one, m_c);
  if (!ok && N < 10) {
    printf("B-Ce:\n"); matrix_printf(stdout, "%13e", &B);
  }

  matrix_release(&Ce);
  matrix_release(&A);
  matrix_release(&B);
  matrix_release(&B0);

  return ok;
}

/*
 *
 */
int main(int argc, char **argv)
{

  armas_conf_t conf;
  //armas_d_dense_t C, Ct, T, B0, A, B, Ce;

  int ok, opt;
  int N = 33;
  int K = 33;
  int fails = 0;
  int prec = 200;
  int verbose = 0;
  int flags = ARMAS_UPPER;
  int all = 0;
  double cwant = 1e14; // wanted condition number

  while ((opt = getopt(argc, argv, "C:p:vALRT")) != -1) {
    switch (opt) {
    case 'C':
      cwant = strtod(optarg, (char **)0);
      break;
    case 'p':
      prec = atoi(optarg);
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
    K = N;
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
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
