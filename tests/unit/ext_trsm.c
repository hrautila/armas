
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"
//#include <armas/dmatrix.h>
//#include "helper.h"

/*
 * Generate set matrices such that C = A * B ==> B = A.-1*C
 *
 *  UPPER:                       
 *   (b0 b0)  = (a00 a01) (x0)   
 *   (b1 b1)    ( 0   I ) (x1)   
 *
 *   b0 = a00*x0 + a01*x1 = a00*x0 + Dx
 *   b1 = I*x1 = x1
 *
 *   A is UNIT diagonal: 
 *     a00 = 1.0, x0 = Dx  ==> b0 = 2*Dx
 *   A is not UNIT diagonal:
 *     a00 = Dx,  x0 = 1.0 ==> b0 = 2*Dx
 *
 *  LOWER:                 
 *   (b0) = ( I   0 ) (x0)
 *   (b1)   (a10 a11) (x1)
 *
 *   b0 = x0
 *   b1 = a10*x0 + a11*x1 = Dx + a11*x1
 *
 *   A is UNIT diagonal: 
 *     a11 = 1.0, x1 = Dx  ==> b1 = 2*Dx
 *   A is not UNIT diagonal:
 *     a11 = Dx,  x1 = 1.0 ==> b1 = 2*Dx
 *
 */
void ep_gentrsm(double *dot, double *tcond,
                __Matrix *A, __Matrix *B, __Matrix *C,
                double cond, int flags)
{
  __Matrix R0, C0, C1, D, Rx, Cx;
  int k, tk;
  int right = flags & ARMAS_RIGHT;

  matrix_init(&R0, 0, 0);
  matrix_init(&C0, 0, 0);
  // make A identity
  matrix_set_values(A, zero, 0);
  matrix_diag(&D, A, 0);
  matrix_set_values(&D, one, 0);
  
  switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
  case ARMAS_LOWER|ARMAS_RIGHT:
    matrix_column(&R0, A, 0);
    matrix_subvector(&Rx, &R0, 1, matrix_size(&R0)-1);
    tk = 0;
    break;
  case ARMAS_LOWER|ARMAS_TRANS:
    matrix_column(&R0, A, 0);
    matrix_subvector(&Rx, &R0, 1, matrix_size(&R0)-1);
    tk = 0;
    break;
  case ARMAS_LOWER|ARMAS_TRANS|ARMAS_RIGHT:
    matrix_row(&R0, A, A->rows-1);
    matrix_subvector(&Rx, &R0, 0, matrix_size(&R0)-1);
    tk = B->cols - 1;
    break;
  case ARMAS_LOWER:
    matrix_row(&R0, A, A->rows-1);
    matrix_subvector(&Rx, &R0, 0, matrix_size(&R0)-1);
    tk = B->rows - 1;
    break;
  case ARMAS_UPPER|ARMAS_TRANS:
    matrix_column(&R0, A, A->cols-1);
    matrix_subvector(&Rx, &R0, 0, matrix_size(&R0)-1);
    tk = B->rows - 1;
    break;
  case ARMAS_UPPER|ARMAS_TRANS|ARMAS_RIGHT:
    matrix_row(&R0, A, 0);
    matrix_subvector(&Rx, &R0, 1, matrix_size(&R0)-1);
    tk = 0;
    break;
  case ARMAS_UPPER|ARMAS_RIGHT:
    matrix_column(&R0, A, A->cols-1);
    matrix_subvector(&Rx, &R0, 0, matrix_size(&R0)-1);
    tk = B->cols - 1;
    break;
  case ARMAS_UPPER:
  default:
    matrix_row(&R0, A, 0);
    matrix_subvector(&Rx, &R0, 1, matrix_size(&R0)-1);
    tk = 0;
    break;
  }

  if (right) {
    matrix_row(&C0, B, 0);
  } else {
    matrix_column(&C0, B, 0);
  }
  // generate dot product ...
  matrix_subvector(&Cx, &C0, tk == 0 ? 1 : 0, matrix_size(&C0)-1);
  ep_gendot(dot, tcond, &Rx, &Cx, cond);

  if (flags & ARMAS_UNIT) {
    matrix_set(A, tk, tk, 1.0);
    matrix_set_at(&C0, tk, *dot);
  } else {
    matrix_set(A, tk, tk, *dot);
    matrix_set_at(&C0, tk, 1.0);
  }

  // make rest of rows/columns copies of first row/column.
  for (k = 1; k < (right ? B->rows : B->cols); k++) {
    if (right) {
      matrix_row(&C1, B, k);
    } else {
      matrix_column(&C1, B, k);
    }
    matrix_mcopy(&C1, &C0);
  }

  // create result matrix C; C = A*B where elements row/column at index tk
  // have the value of 2*(dot product).
  matrix_mcopy(C, B);
  if (right) {
    matrix_column(&C1, C, tk);
  } else {
    matrix_row(&C1, C, tk);
  }
  //printf("..gentrsm tk=%d, dot=%13e, C1=[%ld]\n", tk, *dot, matrix_size(&C1));
  for (k = 0; k < matrix_size(&C1); k++) {
    matrix_set_at(&C1, k, 2.0*(*dot));
  }
}

// ne = norm1 of exact; re = ||exact-result||/||exact||
void compute(double *ne, double *re,
             __Matrix *B, __Matrix *A, __Matrix *C,
             int flags, int prec, int verbose, armas_conf_t *conf)
{
  __Dtype e;
  // B = A*B
  matrix_solve_trm(B, A, 1.0, flags, conf);
  if (verbose > 1 && B->cols < 10) {
    printf("A.-1*B:\n"); matrix_printf(stdout, "%13e", B);
  }
  *re = rel_error(&e, B, C, ARMAS_NORM_INF, ARMAS_NONE, conf);
  if (ne)
    *ne = e;
}

int test(char *name, int N, int K, int flags, int verbose, int prec, double cwant, armas_conf_t *conf)
{
  __Matrix B0, A, B, Ce;
  double dot, cond, m_c, m_one;
  int ok;


  matrix_init(&A, N, N);
  if (flags & ARMAS_RIGHT) {
    matrix_init(&Ce, K, N);
    matrix_init(&B, K, N);
    matrix_init(&B0, K, N);
  } else {
    matrix_init(&Ce, N, K);
    matrix_init(&B, N, K);
    matrix_init(&B0, N, K);
  }
  
  ep_gentrsm(&dot, &cond, &A, &B0, &B, cwant, flags);

  if (verbose > 1 && N < 10) {
    printf("A:\n"); matrix_printf(stdout, "%13e", &A);
    if (K < 10) {
      printf("B0:\n"); matrix_printf(stdout, "%13e", &B0);
      printf("B:\n"); matrix_printf(stdout, "%13e", &B);
    }
  }

  // extended precision computations

  // 1. A*B
  compute(&m_c, &m_one, &B, &A, &B0, flags, prec, verbose, conf);

  ok = m_one < N*_EPS;
  printf("%-4s: %s rel.error %e [%e]\n",  PASS(ok), name, m_one, m_c);
  if (!ok && K < 10) {
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
  int N = 33;
  int K = 33;
  int fails = 0;
  int normal_prec = 0;
  int prec = 200;
  int verbose = 0;
  int flags = ARMAS_UPPER;
  int all = 1;
  int unit = 0;
  int naive = 0;
  double cwant = 1.0/_EPS; // wanted condition number

  while ((opt = getopt(argc, argv, "C:p:vnASsLRTU")) != -1) {
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
    case 'R':
      flags &= ~ARMAS_LEFT;
      flags |= ARMAS_RIGHT;
      break;
    case 'T':
      flags |= ARMAS_TRANS;
      break;
    case 'U':
      flags |= ARMAS_UNIT;
      unit = ARMAS_UNIT;
      break;
    default:
      fprintf(stderr, "usage: ext_trsm [-p bits -C cond -v -SLRT] [size]\n");
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
  if (!normal_prec)
    conf.optflags |= ARMAS_OEXTPREC;
  if (naive)
    conf.optflags |= ARMAS_ONAIVE;

  if (!all) {
    //printf("conf .mb, .nb, .kb: %d, %d, %d\n", conf.mb, conf.nb, conf.kb);
    if (flags & ARMAS_UNIT)
      printf("UNIT matrix A...\n");
    ok = test("single", N, K, flags, verbose, prec, cwant, &conf);
    fails += 1 - ok;
  } else {
    ok = test("u(A)*B  ", N, K, unit|ARMAS_UPPER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("l(A)*B  ", N, K, unit|ARMAS_LOWER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("u(A.T)*B", N, K, unit|ARMAS_UPPER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("l(A.T)*B", N, K, unit|ARMAS_LOWER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*u(A)  ", N, K, unit|ARMAS_RIGHT|ARMAS_UPPER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*l(A)  ", N, K, unit|ARMAS_RIGHT|ARMAS_LOWER, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*u(A.T)", N, K, unit|ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
    ok = test("B*l(A.T)", N, K,unit| ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANS, verbose, prec, cwant, &conf);
    fails += 1 - ok;
  }

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
