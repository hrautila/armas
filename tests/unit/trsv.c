
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"
//#include <armas/dmatrix.h>
//#include "helper.h"

main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t X, Y, X0, A, At;

  int ok, opt;
  int N = 611;
  int flags = 0;
  int nproc = 1;

  while ((opt = getopt(argc, argv, "a:t:T:p:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: trsv [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc) {
     N = atoi(argv[optind]);
  }

  conf.mb = 64;
  conf.nb = 96;
  conf.kb = 160;
  conf.maxproc = 1;

  int fails = 0;

  armas_d_init(&Y, N, 1);
  armas_d_init(&X0, N, 1);
  armas_d_init(&X, N, 1);
  armas_d_init(&A, N, N);
  armas_d_init(&At, N, N);
  
  armas_d_set_values(&X0, one, ARMAS_NULL);
  armas_d_set_values(&X, one, ARMAS_NULL);
  armas_d_set_values(&A, zeromean, ARMAS_UPPER);
  armas_d_set_values(&Y, zero, ARMAS_NULL);
  armas_d_transpose(&At, &A);

  // X = A*X
  armas_d_mvmult_trm(&X, &A, 1.0, ARMAS_UPPER, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, ARMAS_UPPER, &conf);
  ok = armas_d_allclose(&X0, &X);
  printf("%6s : X = trsv(trmv(X, A, U|N), A, U|N)\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  armas_d_set_values(&X, one, ARMAS_NULL);
  // X = A*X
  armas_d_mvmult_trm(&X, &A, 1.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, ARMAS_UPPER|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&X0, &X);
  printf("%6s : X = trsv(trmv(X, A, U|T), A, U|T)\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  armas_d_set_values(&A, zero, ARMAS_NULL);
  armas_d_set_values(&A, zeromean, ARMAS_LOWER);
  armas_d_set_values(&X, one, ARMAS_NULL);

  // X = A*X
  armas_d_mvmult_trm(&X, &A, 1.0, ARMAS_LOWER, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, ARMAS_LOWER, &conf);
  ok = armas_d_allclose(&X0, &X);
  printf("%6s : X = trsv(trmv(X, A, L|N), A, L|N)\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  armas_d_set_values(&X, one, ARMAS_NULL);
  // X = A*X
  armas_d_mvmult_trm(&X, &A, 1.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);
  armas_d_mvsolve_trm(&X, &A, 1.0, ARMAS_LOWER|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&X0, &X);
  printf("%6s : X = trsv(trmv(X, A, L|T), A, L|T)\n", ok ? "OK" : "FAILED");
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
