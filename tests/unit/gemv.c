
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t X, Y, Y0, A, At;

  int ok, opt;
  int N = 1307;
  int M = 1025;
  int nproc = 1;
  int bsize = 0;
  int psize = 10;
  int algo = 'B';

  while ((opt = getopt(argc, argv, "a:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: gemv [-P nproc] [size]\n");
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
  conf.maxproc = 1;

  armas_d_init(&Y, M, 1);
  armas_d_init(&Y0, M, 1);
  armas_d_init(&X, N, 1);
  armas_d_init(&A, M, N);
  armas_d_init(&At, N, M);
  
  armas_d_set_values(&Y, zero, ARMAS_NULL);
  armas_d_set_values(&Y0, zero, ARMAS_NULL);
  armas_d_set_values(&X, unitrand, ARMAS_NULL);
  armas_d_set_values(&A, unitrand, ARMAS_NULL);
  armas_d_transpose(&At, &A);

  // Y = A*X
  armas_d_mvmult(&Y, &A, &X, 1.0, 0.0, 0, &conf);
  // Y = Y - A*X
  armas_d_mvmult(&Y, &At, &X, -1.0, 1.0, ARMAS_TRANS, &conf);
  ok = armas_d_allclose(&Y, &Y0);
  printf("%6s : gemv(A, X) == gemv(A.T, X, T)\n", ok ? "OK" : "FAILED");
  exit(1 - ok);

}

// Local Variables:
// indent-tabs-mode: nil
// End:
