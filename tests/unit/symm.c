
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  int ok, opt;
  armas_conf_t conf;
  armas_d_dense_t C, C0, C1, A, As,  B;
  int nsched = 4;
  int N = 633;
  int nproc = 1;
  int bsize = 0;

  while ((opt = getopt(argc, argv, "P:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: symm [-P nproc] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  conf.mb = 64; conf.nb = 128; conf.kb = 160;
  conf.maxproc = 2;
  conf.optflags = 0;

  armas_d_init(&C0, N, N);
  armas_d_init(&C1, N, N);
  armas_d_init(&A, N, N);
  armas_d_init(&As, N, N);
  armas_d_init(&B, N, N);
  
  armas_d_set_values(&C0, zero, ARMAS_NULL);
  armas_d_set_values(&C1, zero, ARMAS_NULL);
  armas_d_set_values(&A, unitrand, ARMAS_SYMM);
  armas_d_set_values(&B, unitrand, ARMAS_NULL);

  // C0 = symm(upper(A)*B);  C1 = A*B
  armas_d_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_LEFT|ARMAS_UPPER, &conf);
  armas_d_mult(&C1, &A, &B, 1.0, 0.0, 0, &conf);
  ok = armas_d_allclose(&C0, &C1);
  printf("%6s: symm(upper(A), B) == gemm(A, B)\n", ok ? "OK" : "FAILED");
  if (! ok)
    exit(1 - ok);

  // C = B*upper(A)
  armas_d_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_RIGHT|ARMAS_UPPER, &conf);
  armas_d_mult(&C1, &B, &A, 1.0, 0.0, 0, &conf);
  ok = armas_d_allclose(&C0, &C1);
  printf("%6s: symm(B, upper(A)) == gemm(B, A)\n", ok ? "OK" : "FAILED");
  if (!ok)
    exit(1-ok);

  // LOWER
  armas_d_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_LEFT|ARMAS_LOWER, &conf);
  armas_d_mult(&C1, &A, &B, 1.0, 0.0, 0, &conf);
  ok = armas_d_allclose(&C0, &C1);
  printf("%6s: symm(lower(A), B) == gemm(A, B)\n", ok ? "OK" : "FAILED");
  if (! ok)
    exit(1 - ok);

  // C = B*upper(A)
  armas_d_mult_sym(&C0, &A, &B, 1.0, 0.0, ARMAS_RIGHT|ARMAS_LOWER, &conf);
  armas_d_mult(&C1, &B, &A, 1.0, 0.0, 0, &conf);
  ok = armas_d_allclose(&C0, &C1);
  printf("%6s: symm(B, lower(A)) == gemm(B, A)\n", ok ? "OK" : "FAILED");
  exit(1 - ok);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
