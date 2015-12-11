
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#if 0
#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t __Matrix ;
typedef float __Dtype;

#define matrix_init       armas_s_init
#define matrix_set_values armas_s_set_values
#define matrix_mult       armas_s_mult
#define matrix_transpose  armas_s_transpose
#define matrix_release    armas_s_release
#define matrix_printf     armas_s_printf
#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define matrix_init       armas_d_init
#define matrix_set_values armas_d_set_values
#define matrix_mult       armas_d_mult
#define matrix_transpose  armas_d_transpose
#define matrix_release    armas_d_release
#define matrix_printf     armas_d_printf

#endif
#include "helper.h"
#endif
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
int main(int argc, char **argv)
{

  armas_conf_t conf;
  __Matrix A, B, C, Ct, T; //, B, A;

  int ok, opt;
  int N = 633;
  int M = 653;
  int K = 337;
  int verbose = 0;
  int fails = 0;
  __Dtype n0, n1;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose += 1;
      break;
    default:
      fprintf(stderr, "usage: test_symm [-v] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc) {
    N = atoi(argv[optind]);
    K = N;
    M = N;
  }

  conf = *armas_conf_default();

  matrix_init(&C, M, N);
  matrix_init(&Ct, N, M);
  matrix_init(&T, M, N);
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_set_values(&Ct, zero, ARMAS_NULL);

  // test 1: M != N != K
  matrix_init(&A, M, K);
  matrix_init(&B, K, N);
  matrix_set_values(&A, unitrand, ARMAS_NULL);
  matrix_set_values(&B, unitrand, ARMAS_NULL);

  // C = A*B; C.T = B.T*A.T
  matrix_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
  matrix_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);

  matrix_transpose(&T, &Ct);

  n0 = rel_error(&n1, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);

  // accept zero error
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: gemm(A, B)   == transpose(gemm(B.T, A.T))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // test 2: M != N == K
  matrix_set_values(&Ct, zero, ARMAS_NULL);
  matrix_release(&A);
  matrix_release(&B);
  matrix_init(&A, M, N);
  matrix_init(&B, N, N);
  matrix_set_values(&A, unitrand, ARMAS_NULL);
  matrix_set_values(&B, unitrand, ARMAS_NULL);
  // C = A*B.T; Ct = B*A.T
  matrix_mult(&C,  &A, &B, 1.0, 0.0, ARMAS_TRANSB, &conf);
  matrix_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSB, &conf);
  matrix_transpose(&T, &Ct);

  n0 = rel_error((__Dtype *)0, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);

  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: gemm(A, B.T) == transpose(gemm(B, A.T))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // test 3: M == K != N
  matrix_set_values(&Ct, zero, ARMAS_NULL);
  matrix_release(&A);
  matrix_release(&B);
  matrix_init(&A, M, M);
  matrix_init(&B, M, N);
  matrix_set_values(&A, unitrand, ARMAS_NULL);
  matrix_set_values(&B, unitrand, ARMAS_NULL);
  // C = A.T*B; Ct = B.T*A
  matrix_mult(&C,  &A, &B, 1.0, 0.0, ARMAS_TRANSA, &conf);
  matrix_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA, &conf);
  matrix_transpose(&T, &Ct);

  n0 = rel_error((__Dtype *)0, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);

  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: gemm(A.T, B) == transpose(gemm(B.T, A))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // test 1: M != N != K
  matrix_init(&A, M, K);
  matrix_init(&B, K, N);
  matrix_set_values(&A, unitrand, ARMAS_NULL);
  matrix_set_values(&B, unitrand, ARMAS_NULL);

  // C = A*B; C.T = B.T*A.T
  matrix_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
  matrix_mscale(&A, -1.0, 0);
  matrix_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA|ARMAS_TRANSB|ARMAS_ABSB, &conf);
  matrix_transpose(&T, &Ct);

  n0 = rel_error(&n1, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: gemm(A, B)   == transpose(gemm(B.T, |-A.T|))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
