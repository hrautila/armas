
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

/*
 *  Generate set matrices such that C = A * B
 *
 *   (D0 D0 D0)    (q0 q1 q2 q3)  (z0 z0 z0)
 *   (z1 z1 z1) =  ( 0  1  0  0)  (z1 z1 z1)
 *   (z2 z2 z2)    ( 0  0  1  0)  (z2 z2 z2)
 *   (z3 z3 z3)    ( 0  0  0  1)  (z3 z3 z3)
 *
 *   D0 = (q0 q1 q2 q3)*(z0 z1 z2 z3).T   (dot product)
 */
// C = A*B  => B = A.-1*C
void ep_gentrsm(double *dot, double *tcond,
                armas_d_dense_t *A, armas_d_dense_t *B, armas_d_dense_t *C,
                double cond, int flags)
{
    armas_d_dense_t R0, C0, R1, C1, D;
    int k, tk;
    int right = flags & ARMAS_RIGHT;

    // make A identity
    armas_d_set_values(A, zero, 0);
    armas_d_diag(&D, A, 0);
    armas_d_set_values(&D, one, 0);
    
    switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_LOWER|ARMAS_RIGHT:
        armas_d_column(&R0, A, 0);
        tk = 0;
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        armas_d_column(&R0, A, 0);
        tk = 0;
        break;
    case ARMAS_LOWER|ARMAS_TRANS|ARMAS_RIGHT:
        armas_d_row(&R0, A, A->rows-1);
        tk = B->cols - 1;
        break;
    case ARMAS_LOWER:
        armas_d_row(&R0, A, A->rows-1);
        tk = B->rows - 1;
        break;
    case ARMAS_UPPER|ARMAS_TRANS:
        armas_d_column(&R0, A, A->cols-1);
        tk = B->rows - 1;
        break;
    case ARMAS_UPPER|ARMAS_TRANS|ARMAS_RIGHT:
        armas_d_row(&R0, A, 0);
        tk = 0;
        break;
    case ARMAS_UPPER|ARMAS_RIGHT:
        armas_d_column(&R0, A, A->cols-1);
        tk = B->cols - 1;
        break;
    case ARMAS_UPPER:
    default:
        armas_d_row(&R0, A, 0);
        tk = 0;
        break;
    }

    if (right) {
        armas_d_row(&C0, B, 0);
    } else {
        armas_d_column(&C0, B, 0);
    }
    // generate dot product ...
    ep_gendot(dot, tcond, &R0, &C0, cond);

     // make rest of rows/columns copies of first row/column.
    for (k = 1; k < (right ? B->rows : B->cols); k++) {
        if (right) {
            armas_d_row(&C1, B, k);
        } else {
            armas_d_column(&C1, B, k);
        }
        armas_d_mcopy(&C1, &C0);
    }

    // create result matrix C; C = A*B where elements row/column at index tk
    // have the value of dot product.
    armas_d_mcopy(C, B);
    if (right) {
      armas_d_column(&C1, C, tk);
    } else {
      armas_d_row(&C1, C, tk);
    }
    printf("..gentrsm tk=%d, dot=%13e, C1=[%ld]\n", tk, *dot, armas_d_size(&C1));
    for (k = 0; k < armas_d_size(&C1); k++) {
      armas_d_set_at(&C1, k, *dot);
    }
}

// ne = norm1 of exact; re = norm1 of exact-result
void compute(double *ne, double *re,
             armas_d_dense_t *B, armas_d_dense_t *A, armas_d_dense_t *C,
             int flags, int prec, int verbose, armas_conf_t *conf)
{
  int bits = 0;


  // B = A*B
  armas_d_solve_trm(B, A, 1.0, flags, conf);
  if (verbose > 1 && A->rows < 10) {
    //printf("exc(A*B):\n"); armas_d_printf(stdout, "%13e", C);
    printf("A.-1*B:\n"); armas_d_printf(stdout, "%13e", B);
  }
  armas_d_scale_plus(B, C, 1.0, -1.0, ARMAS_NONE, conf);

  *re = armas_d_mnorm(B, ARMAS_NORM_ONE, conf);
}

int test(char *name, int N, int K, int flags, int verbose, int prec, double cwant, armas_conf_t *conf)
{
  armas_d_dense_t C, Ct, T, B0, A, B, Ce;
  double dot, cond, m_c, m_one;
  int ok;


  armas_d_init(&A, N, N);
  if (flags & ARMAS_RIGHT) {
    armas_d_init(&Ce, K, N);
    armas_d_init(&B, K, N);
    armas_d_init(&B0, K, N);
  } else {
    armas_d_init(&Ce, N, K);
    armas_d_init(&B, N, K);
    armas_d_init(&B0, N, K);
  }
  
  ep_gentrsm(&dot, &cond, &A, &B0, &B, cwant, flags);

  if (verbose > 1 && N < 10) {
    printf("A:\n"); armas_d_printf(stdout, "%13e", &A);
    if (K < 10) {
      printf("B0:\n"); armas_d_printf(stdout, "%13e", &B0);
      printf("B:\n"); armas_d_printf(stdout, "%13e", &B);
    }
  }

  // extended precision computations

  // 1. A*B
  compute(&m_c, &m_one, &B, &A, &B0, flags, prec, verbose, conf);

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
  int naive = 0;
  double cwant = 1e14; // wanted condition number
  double dot, cond, m_one_t, m_one, m_c;

  while ((opt = getopt(argc, argv, "C:p:vnASLRT")) != -1) {
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
    case 'n':
      naive = 1;
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
  if (!normal_prec)
    conf.optflags |= ARMAS_OEXTPREC;
  if (naive)
    conf.optflags |= ARMAS_ONAIVE;

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
