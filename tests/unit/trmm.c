
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, C0, B0, A, B;

  int ok, opt, err;
  int N = 600;
  int nproc;
  double alpha = 1.0;

  while ((opt = getopt(argc, argv, "a:s:t:T:P:B:C:p:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: test_trmm [-P nproc] [size]\n");
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

  armas_d_init(&C, N, N);
  armas_d_init(&A, N, N);
  armas_d_init(&B, N, N);
  armas_d_init(&B0, N, N);
  
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_set_values(&A, zero, ARMAS_NULL);
  armas_d_set_values(&A, unitrand, ARMAS_UPPER);
  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);

  // B = A*B
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  armas_d_mult(&C, &A, &B0, alpha, 0.0, ARMAS_NULL, &conf);
  ok = armas_d_allclose(&C, &B);
  printf("%6s: trmm(B, A, L|U)   == gemm(TriU(A), B)\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_set_values(&B0, one, ARMAS_NULL);
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  armas_d_mult(&C, &A, &B0, alpha, 0.0, ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&C, &B);
  printf("%6s: trmm(B, A, L|U|T) == gemm(TriU(A).T, B)\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_set_values(&B0, one, ARMAS_NULL);
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  armas_d_mult(&C, &B0, &A, alpha, 0.0, ARMAS_NULL, &conf);
  ok = armas_d_allclose(&C, &B);
  printf("%6s: trmm(B, A, R|U)   == gemm(B, TriU(A))\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_set_values(&B0, one, ARMAS_NULL);
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  armas_d_mult(&C, &B0, &A, alpha, 0.0, ARMAS_TRANSB, &conf);
  ok = armas_d_allclose(&C, &B);
  printf("%6s: trmm(B, A, R|U|T) == gemm(B, TriU(A).T)\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  exit(0);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
