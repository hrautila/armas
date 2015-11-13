
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"

/*
 * See comments in ext_trsm.c
 */
void ep_gentrsv(double *dot, double *tcond,
                __Matrix *A, __Matrix *X, __Matrix *Y,
                double cond, int flags)
{
  __Matrix R0, D, Rx, Cx;
  int tk;

  // make A identity
  matrix_init(&R0, 0, 0);
  matrix_set_values(A, zero, 0);
  matrix_diag(&D, A, 0);
  matrix_set_values(&D, one, 0);
    
  switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
  case ARMAS_LOWER|ARMAS_TRANS:
    matrix_column(&R0, A, 0);
    matrix_subvector(&Rx, &R0, 1, matrix_size(&R0)-1);
    tk = 0;
    break;
  case ARMAS_LOWER:
    matrix_row(&R0, A, A->rows-1);
    matrix_subvector(&Rx, &R0, 0, matrix_size(&R0)-1);
    tk = X->rows - 1;
    break;
  case ARMAS_UPPER|ARMAS_TRANS:
    matrix_column(&R0, A, A->cols-1);
    matrix_subvector(&Rx, &R0, 0, matrix_size(&R0)-1);
    tk = X->rows - 1;
    break;
  case ARMAS_UPPER:
  default:
    matrix_row(&R0, A, 0);
    matrix_subvector(&Rx, &R0, 1, matrix_size(&R0)-1);
    tk = 0;
    break;
  }

  // generate dot product ...
  matrix_subvector(&Cx, X, tk == 0 ? 1 : 0, matrix_size(X)-1);
  ep_gendot(dot, tcond, &Rx, &Cx, cond);
  if (flags & ARMAS_UNIT) {
    matrix_set(A, tk, tk, 1.0);
    matrix_set_at(X, tk, *dot);
  } else {
    matrix_set(A, tk, tk, *dot);
    matrix_set_at(X, tk, 1.0);
  }
  
  // make result vector
  matrix_mcopy(Y, X);
  matrix_set_at(Y, tk, 2.0*(*dot));
}

// ne = norm1 of exact; re = ||exact-result||/||exact||
void compute(double *ne, double *re,
             __Matrix *B, __Matrix *A, __Matrix *C,
             int flags, int prec, int verbose, armas_conf_t *conf)
{
  // B = A*B
  matrix_mvsolve_trm(B, A, 1.0, flags, conf);
  if (verbose > 1 && A->rows < 10) {
    printf("A.-1*B:\n"); matrix_printf(stdout, "%13e", B);
  }
  *re = rel_error(ne, B, C, ARMAS_NORM_INF, ARMAS_NONE, conf);
}

int test(char *name, int N, int flags, int verbose, int prec, double cwant, armas_conf_t *conf)
{
  __Matrix B0, A, B, Ce;
  double dot, cond, m_c, m_rel;
  int ok;


  matrix_init(&A, N, N);
  matrix_init(&Ce, N, 1);
  matrix_init(&B, N, 1);
  matrix_init(&B0, N, 1);
  
  ep_gentrsv(&dot, &cond, &A, &B0, &B, cwant, flags);

  if (verbose > 1 && N < 10) {
    printf("A:\n"); matrix_printf(stdout, "%13e", &A);
    printf("B0:\n"); matrix_printf(stdout, "%13e", &B0);
    printf("B:\n"); matrix_printf(stdout, "%13e", &B);
  }

  // extended precision computations

  // 1. A*B
  compute(&m_c, &m_rel, &B, &A, &B0, flags, prec, verbose, conf);

  ok = m_rel < N*_EPS;
  printf("%-4s: %s rel.error %e [%e]\n",  PASS(ok), name, m_rel, m_c);
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

  int ok, opt;
  int N = 121;
  int fails = 0;
  int normal_prec = 0;
  int prec = 200;
  int verbose = 0;
  int flags = ARMAS_UPPER;
  int all = 1;
  int unit = 0;
  int naive = 0;
  double cwant = 1.0/_EPS; // wanted condition number

  while ((opt = getopt(argc, argv, "C:p:vnSsLTU")) != -1) {
    switch (opt) {
    case 'C':
      cwant = strtod(optarg, (char **)0);
      break;
    case 'p':
      prec = atoi(optarg);
      break;
    case 's':
      normal_prec = 1;
      break;
    case 'n':
      naive = 1;
      break;
    case 'v':
      verbose += 1;
      break;
    case 'S':
      all = 0;
      break;
    case 'L':
      flags &= ~ARMAS_UPPER;
      flags |= ARMAS_LOWER;
      break;
    case 'T':
      flags |= ARMAS_TRANS;
      break;
    case 'U':
      flags |= ARMAS_UNIT;
      unit  |= ARMAS_UNIT;
      break;
    default:
      fprintf(stderr, "usage: ext_trsv [-p bits -C cond -v -SLRTU] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc) {
    N = atoi(argv[optind]);
  }
  
  conf = *armas_conf_default();
  if (!normal_prec)
    conf.optflags |= ARMAS_OEXTPREC;
  if (naive)
    conf.optflags |= ARMAS_ONAIVE;

  if (!all) {
    //printf("conf .mb, .nb, .kb: %d, %d, %d\n", conf.mb, conf.nb, conf.kb);
    ok = test("single", N, flags, verbose, prec, cwant, &conf);
    fails += 1 - ok;
  } else {
    ok = test("N: u(A)*B  ", N, ARMAS_UPPER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("N: l(A)*B  ", N, ARMAS_LOWER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("N: u(A.T)*B", N, ARMAS_UPPER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("N: l(A.T)*B", N, ARMAS_LOWER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;

    ok = test("U: u(A)*B  ", N, ARMAS_UNIT|ARMAS_UPPER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("U: l(A)*B  ", N, ARMAS_UNIT|ARMAS_LOWER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("U: u(A.T)*B", N, ARMAS_UNIT|ARMAS_UPPER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("U: l(A.T)*B", N, ARMAS_UNIT|ARMAS_LOWER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
  }

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
