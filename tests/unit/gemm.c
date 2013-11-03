
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, Ct, T, C0, A, B;

  int ok, opt;
  int N = 633;
  int nproc = 1;
  int fails = 0;
  uint64_t pschedule[] = {6*6, 10*10, 600*600, 800*800};
  int nsched = 4;

  while ((opt = getopt(argc, argv, "P:B:p:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: test_symm [-P nproc] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf.mb = 64;
  conf.nb = 96;
  conf.kb = 160;
  conf.maxproc = nproc;
  conf.optflags = 0;

  armas_d_init(&C, N, N);
  armas_d_init(&Ct, N, N);
  armas_d_init(&T, N, N);

  armas_d_init(&A, N, N);
  armas_d_init(&B, N, N);
  
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_set_values(&Ct, zero, ARMAS_NULL);
  armas_d_set_values(&A, unitrand, ARMAS_NULL);
  armas_d_set_values(&B, unitrand, ARMAS_NULL);

  // C = A*B; C.T = B.T*A.T
  armas_d_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
  armas_d_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA|ARMAS_TRANSB, &conf);
  armas_d_transpose(&T, &Ct);
  ok = armas_d_allclose(&C, &T);
  printf("%6s: gemm(A, B)   == transpose(gemm(B.T, A.T))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // C = A*B.T; Ct = B*A.T
  armas_d_mult(&C,  &A, &B, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_d_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSB, &conf);
  armas_d_transpose(&T, &Ct);
  ok = armas_d_allclose(&C, &T);
  printf("%6s: gemm(A, B.T) == transpose(gemm(B, A.T))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  // C = A.T*B; Ct = B.T*A
  armas_d_mult(&C,  &A, &B, 1.0, 0.0, ARMAS_TRANSA, &conf);
  armas_d_mult(&Ct, &B, &A, 1.0, 0.0, ARMAS_TRANSA, &conf);
  armas_d_transpose(&T, &Ct);
  ok = armas_d_allclose(&C, &T);
  printf("%6s: gemm(A.T, B) == transpose(gemm(B.T, A))\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
