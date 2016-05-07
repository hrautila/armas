
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t __Matrix ;
typedef float __Dtype;

#define matrix_init       armas_s_init
#define matrix_set_values armas_s_set_values
#define matrix_mult       armas_s_mult
#define matrix_transpose  armas_s_transpose
#define matrix_release    armas_s_release
#define matrix_make_trm   armas_s_make_trm
#define matrix_update_trm armas_s_update_trm

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define matrix_init       armas_d_init
#define matrix_set_values armas_d_set_values
#define matrix_mult       armas_d_mult
#define matrix_transpose  armas_d_transpose
#define matrix_release    armas_d_release
#define matrix_make_trm   armas_d_make_trm
#define matrix_update_trm armas_d_update_trm

#endif
#include "helper.h"

int main(int argc, char **argv)
{
  armas_conf_t conf;
  __Matrix C, C0, A, At, B, Bt;

  int ok, opt;
  int N = 311, M = 353;
  int verbose = 1;
  int fails = 0;
  int algo = 'B';
  __Dtype n0, n1, alpha = 1.0;

  while ((opt = getopt(argc, argv, "va:")) != -1) {
    switch (opt) {
    case 'v':
      verbose++;
      break;
    case 'a':
      algo = *optarg;
      break;
    default:
      fprintf(stderr, "usage: trmupd [-P nproc] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc-1) {
    M = atoi(argv[optind]);
    N = atoi(argv[optind+1]);
  } else if (optind < argc) {
    M = N = atoi(argv[optind]);
  }

  conf = *armas_conf_default();
  switch (algo) {
  case 'r':
  case 'R':
    conf.optflags |= ARMAS_ORECURSIVE;
    break;
  case 'n':
  case 'N':
    conf.optflags |= ARMAS_ONAIVE;
    break;
  }

  matrix_init(&C,  M, N);
  matrix_init(&C0, M, N);
  matrix_init(&A,  M, N/2);
  matrix_init(&B,  N/2, N);
  matrix_init(&At, N/2, M);
  matrix_init(&Bt, N, N/2);
  
  matrix_set_values(&A, zeromean, ARMAS_NULL);
  matrix_set_values(&B, zeromean, ARMAS_NULL);
  matrix_transpose(&At, &A);
  matrix_transpose(&Bt, &B);

  printf("C(M,N)  M > N: M=%d, N=%d, K=%d\n", M, N, N/2);
  // upper(C)
  matrix_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  matrix_make_trm(&C0, ARMAS_UPPER);
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_make_trm(&C, ARMAS_UPPER);

  // ----------------------------------------------------------------------------
  // 1. C = upper(C) + A*B
  matrix_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_UPPER, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|N|N) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = upper(C) + A.T*B
  matrix_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|T|N) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = upper(C) + A*B.T
  matrix_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSB, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|N|T) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = upper(C) + A.T*B.T
  matrix_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|T|T) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // lower(C)
  matrix_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  matrix_make_trm(&C0, ARMAS_LOWER);
  matrix_make_trm(&C, ARMAS_LOWER);

  // ----------------------------------------------------------------------------
  // 1. C = lower(C) + A*B
  matrix_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_LOWER, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|N|N) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = lower(C) + A.T*B
  matrix_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|T|N) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = lower(C) + A*B.T
  matrix_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSB, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|N|T) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = lower(C) + A.T*B.T
  matrix_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|T|T) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // reallocate 
  matrix_release(&A);
  matrix_release(&B);
  matrix_release(&C);
  matrix_release(&C0);
  matrix_release(&At);
  matrix_release(&Bt);

  int t = N;
  N = M; M = t;
  matrix_init(&C,  M, N);
  matrix_init(&C0, M, N);
  matrix_init(&A,  M, N/2);
  matrix_init(&B,  N/2, N);
  matrix_init(&At, N/2, M);
  matrix_init(&Bt, N, N/2);
  
  matrix_set_values(&A, zeromean, ARMAS_NULL);
  matrix_set_values(&B, zeromean, ARMAS_NULL);
  matrix_transpose(&At, &A);
  matrix_transpose(&Bt, &B);

  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  matrix_make_trm(&C0, ARMAS_UPPER);

  printf("C(M,N)  M < N: M=%d, N=%d, K=%d\n", M, N, N/2);

  // ----------------------------------------------------------------------------
  // 1. C = upper(C) + A*B
  matrix_make_trm(&C, ARMAS_UPPER);
  matrix_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_UPPER, &conf);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|N|N) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = upper(C) + A.T*B
  matrix_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|T|N) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = upper(C) + A*B.T
  matrix_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSB, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|N|T) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = upper(C) + A.T*B.T
  matrix_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, U|T|T) == TriU(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 1. C = lower(C) + A*B
  matrix_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  matrix_make_trm(&C0, ARMAS_LOWER);

  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_make_trm(&C, ARMAS_LOWER);
  matrix_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_LOWER, &conf);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|N|N) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = lower(C) + A.T*B
  matrix_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|T|N) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = lower(C) + A*B.T
  matrix_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSB, &conf);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|N|T) == TriL(gemm(C, A, B))\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = lower(C) + A.T*B.T
  matrix_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);

  n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
  ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
  printf("%6s: trmupd(C, A, B, L|T|T) == TriL(gemm(C, A, B))\n", PASS(ok));
  fails += 1 - ok;
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
  }

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
