
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include <armas/dmatrix.h>
#include "helper.h"


main(int argc, char **argv) {

  armas_conf_t conf;
  armas_d_dense_t C, B0, A, B;

  int ok, opt, err, k, i;
  int N = 601;
  int nproc = 1;
  int fails = 0;
  double alpha = 1.0;

  while ((opt = getopt(argc, argv, "P:")) != -1) {
    switch (opt) {
    case 'P':
      nproc = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: trsm [-P nproc] [size]\n");
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

  armas_d_init(&A, N, N);
  armas_d_init(&B, N, N);
  armas_d_init(&B0, N, N);
  
  // Upper triangular matrix
  armas_d_set_values(&A, zeromean, ARMAS_UPPER);

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  ok = check(&B, &B0, -1, "B = solve_trm(mult_trm(B, A, L|U|N), A, L|U|N)");
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  ok = check(&B, &B0, 1, "B = solve_trm(mult_trm(B, A, R|U|N), A, R|U|N)");
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  ok = check(&B, &B0, -1, "B = solve_trm(mult_trm(B, A, L|U|T), A, L|U|T)");
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  ok = check(&B, &B0, 1, "B = solve_trm(mult_trm(B, A, R|U|T), A, R|U|T)");
  fails += 1 - ok;

  // Lower triangular matrix
  armas_d_set_values(&A, zeromean, ARMAS_LOWER);

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT, &conf);
  ok = check(&B, &B0, -1, "B = solve_trm(mult_trm(B, A, L|L|N), A, L|L|N)");
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT, &conf);
  ok = check(&B, &B0, 1, "B = solve_trm(mult_trm(B, A, R|L|N), A, R|L|N)");
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  ok = check(&B, &B0, 1, "B = solve_trm(mult_trm(B, A, L|L|T), A, L|L|T)");
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  ok = check(&B, &B0, -1, "B = solve_trm(mult_trm(B, A, R|L|T), A, R|L|T)");
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
