
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, C0, B0, A, At;

  int ok, opt, err;
  int N = 8;
  int right = 0;
  int lower = 0;
  int flags = 0;
  int nproc = 1;
  int trans = 0;
  int bsize = 0;
  int algo = 'B';
  double alpha = 1.0;
  double inv_alpha = 1.0;

  while ((opt = getopt(argc, argv, "P:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: syr2k [-P nproc] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf.mb = 64;
  conf.nb = 96;
  conf.kb = 160;
  conf.optflags = 0;
  conf.maxproc = nproc;

  armas_d_init(&C,  N, N);
  armas_d_init(&C0, N, N);
  armas_d_init(&A,  N, N/2);
  armas_d_init(&At, N/2, N);
  armas_d_init(&B,  N, N/2);
  armas_d_init(&Bt, N/2, N);
  
  armas_d_set_values(&A, zeromean, ARMAS_NULL);
  armas_d_set_values(&B, zeromean, ARMAS_NULL);
  armas_d_transpose(&At, &A);
  armas_d_transpose(&Bt, &B);

  // 1. C = upper(C) + A*B.T + B*A.T;
  armas_d_set_values(&C, one, ARMAS_SYMM);
  armas_d_mcopy(&C0, &C);

  armas_d_mk_trm(&C, ARMAS_UPPER);
  armas_d_2update_sym(&C, &A, &B, alpha, 0.0, ARMAS_UPPER, &conf);

  armas_d_mult(&C0, &A, &At, alpha, 0.0, ARMAS_NULL, &conf);
  armas_d_mk_trm(&C0, ARMAS_UPPER);
  ok = armas_d_allclose(&C0, &C);

  printf("%6s: syrk(C, A, U|N) == TriU(gemm(C, A, A.T))\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  // 2. C = upper(C) + A.T*A
  armas_d_set_values(&C, one, ARMAS_SYMM);
  armas_d_mcopy(&C0, &C);

  armas_d_mk_trm(&C, ARMAS_UPPER);
  armas_d_update_sym(&C, &At, alpha, 0.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);

  armas_d_mult(&C0, &At, &A, alpha, 0.0, ARMAS_TRANSA, &conf);
  armas_d_mk_trm(&C0, ARMAS_UPPER);
  ok = armas_d_allclose(&C0, &C);

  printf("%6s: syrk(C, A, T|N) == TriU(gemm(C, A.T, A))\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  // 1. C = lower(C) + A*A.T;
  armas_d_set_values(&C, one, ARMAS_SYMM);
  armas_d_mcopy(&C0, &C);

  armas_d_mk_trm(&C, ARMAS_LOWER);
  armas_d_update_sym(&C, &A, alpha, 0.0, ARMAS_LOWER, &conf);

  armas_d_mult(&C0, &A, &At, alpha, 0.0, ARMAS_NULL, &conf);
  armas_d_mk_trm(&C0, ARMAS_LOWER);
  ok = armas_d_allclose(&C0, &C);

  printf("%6s: syrk(C, A, L|N) == TriL(gemm(C, A, A.T))\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  // 2. C = lower(C) + A.T*A
  armas_d_set_values(&C, one, ARMAS_SYMM);
  armas_d_mcopy(&C0, &C);

  armas_d_mk_trm(&C, ARMAS_LOWER);
  armas_d_update_sym(&C, &At, alpha, 0.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);

  armas_d_mult(&C0, &At, &A, alpha, 0.0, ARMAS_TRANSA, &conf);
  armas_d_mk_trm(&C0, ARMAS_LOWER);
  ok = armas_d_allclose(&C0, &C);

  printf("%6s: syrk(C, A, T|N) == TriU(gemm(C, A.T, A))\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
