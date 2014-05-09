
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, C0, B0, A, At, B, Bt;

  int ok, opt, err;
  int N = 311, M = 353;
  int nproc = 1;
  int fails = 0;
  int algo = 'B';
  double alpha = 1.0;
  double inv_alpha = 1.0;

  while ((opt = getopt(argc, argv, "P:a:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
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

  conf.mb = 64;
  conf.nb = 96;
  conf.kb = 160;
  conf.optflags = 0;
  conf.maxproc = nproc;
  switch (algo) {
  case 'r':
  case 'R':
    conf.optflags |= ARMAS_RECURSIVE;
    break;
  case 'n':
  case 'N':
    conf.optflags |= ARMAS_SNAIVE;
    break;
  }

  armas_d_init(&C,  M, N);
  armas_d_init(&C0, M, N);
  armas_d_init(&A,  M, N/2);
  armas_d_init(&B,  N/2, N);
  armas_d_init(&At, N/2, M);
  armas_d_init(&Bt, N, N/2);
  
  armas_d_set_values(&A, zeromean, ARMAS_NULL);
  armas_d_set_values(&B, zeromean, ARMAS_NULL);
  armas_d_transpose(&At, &A);
  armas_d_transpose(&Bt, &B);

  printf("C(M,N)  M > N: M=%d, N=%d, K=%d\n", M, N, N/2);
  // upper(C)
  armas_d_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  armas_d_make_trm(&C0, ARMAS_UPPER);
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_make_trm(&C, ARMAS_UPPER);

  // ----------------------------------------------------------------------------
  // 1. C = upper(C) + A*B
  armas_d_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_UPPER, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|N|N) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = upper(C) + A.T*B
  armas_d_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|T|N) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = upper(C) + A*B.T
  armas_d_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|N|T) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = upper(C) + A.T*B.T
  armas_d_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|T|T) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // lower(C)
  armas_d_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  armas_d_make_trm(&C0, ARMAS_LOWER);
  armas_d_make_trm(&C, ARMAS_LOWER);

  // ----------------------------------------------------------------------------
  // 1. C = lower(C) + A*B
  armas_d_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_LOWER, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|N|N) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = lower(C) + A.T*B
  armas_d_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|T|N) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = lower(C) + A*B.T
  armas_d_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|N|T) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = lower(C) + A.T*B.T
  armas_d_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|T|T) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // reallocate 
  armas_d_release(&A);
  armas_d_release(&B);
  armas_d_release(&C);
  armas_d_release(&C0);
  armas_d_release(&At);
  armas_d_release(&Bt);

  int t = N;
  N = M; M = t;
  armas_d_init(&C,  M, N);
  armas_d_init(&C0, M, N);
  armas_d_init(&A,  M, N/2);
  armas_d_init(&B,  N/2, N);
  armas_d_init(&At, N/2, M);
  armas_d_init(&Bt, N, N/2);
  
  armas_d_set_values(&A, zeromean, ARMAS_NULL);
  armas_d_set_values(&B, zeromean, ARMAS_NULL);
  armas_d_transpose(&At, &A);
  armas_d_transpose(&Bt, &B);

  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  armas_d_make_trm(&C0, ARMAS_UPPER);

  printf("C(M,N)  M < N: M=%d, N=%d, K=%d\n", M, N, N/2);

  // ----------------------------------------------------------------------------
  // 1. C = upper(C) + A*B
  armas_d_make_trm(&C, ARMAS_UPPER);
  armas_d_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_UPPER, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|N|N) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = upper(C) + A.T*B
  armas_d_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|T|N) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = upper(C) + A*B.T
  armas_d_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|N|T) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = upper(C) + A.T*B.T
  armas_d_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, U|T|T) == TriU(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 1. C = lower(C) + A*B
  armas_d_mult(&C0, &A, &B, alpha, 0.0, ARMAS_NULL, &conf);
  armas_d_make_trm(&C0, ARMAS_LOWER);

  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_make_trm(&C, ARMAS_LOWER);
  armas_d_update_trm(&C, &A, &B, alpha, 0.0, ARMAS_LOWER, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|N|N) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 2. C = lower(C) + A.T*B
  armas_d_update_trm(&C, &At, &B, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|T|N) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 3. C = lower(C) + A*B.T
  armas_d_update_trm(&C, &A, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|N|T) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // ----------------------------------------------------------------------------
  // 4. C = lower(C) + A.T*B.T
  armas_d_update_trm(&C, &At, &Bt, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C0, &C);
  printf("%6s: trmupd(C, A, B, L|T|T) == TriL(gemm(C, A, B))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;
  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
