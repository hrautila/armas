
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
  double m_inf, m_one;

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
  if (N < 10) {
    printf("A:\n"); armas_d_printf(stdout, "%8.1e", &A);
  }
    

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  if (N < 10) {
    printf("A*B:\n"); armas_d_printf(stdout, "%8.1e", &B);
  }
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT, &conf);
  if (N < 10) {
    printf("A.-1*B:\n"); armas_d_printf(stdout, "%8.1e", &B);
  }
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, L|U|N), A, L|U|N) [_inf=%.3e, _one=%.3e]\n",
         ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, R|U|N), A, R|U|N) [_inf=%.3e, _one=%.3e]\n",
         ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, L|U|T), A, L|U|T) [_inf=%.3e, _one=%.3e]\n",
         ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_UPPER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, R|U|T), A, R|U|T) [_inf=%.3e, _one=%.3e]\n",
    ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  // Lower triangular matrix
  armas_d_set_values(&A, zeromean, ARMAS_LOWER);

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, L|L|N), A, L|L|N) [_inf=%.3e, _one=%.3e]\n",
    ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, R|L|N), A, R|L|N) [_inf=%.3e, _one=%.3e]\n",
    ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_LEFT|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, L|L|T), A, L|L|T) [_inf=%.3e, _one=%.3e]\n",
    ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  armas_d_set_values(&B, one, ARMAS_NULL);
  armas_d_mcopy(&B0, &B);
  armas_d_mult_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  armas_d_solve_trm(&B, &A, alpha, ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA, &conf);
  ok = armas_d_allclose(&B, &B0);
  armas_d_scale_plus(&B, &B0, 1.0, -1.0, ARMAS_NONE, &conf);
  m_inf = armas_d_mnorm(&B, ARMAS_NORM_INF, &conf);
  m_one = armas_d_mnorm(&B, ARMAS_NORM_ONE, &conf);
  printf("%6s: B = solve_trm(mult_trm(B, A, R|L|T), A, R|L|T) [_inf=%.3e, _one=%.3e]\n",
    ok ? "OK" : "FAILED", m_inf, m_one);
  fails += 1 - ok;

  exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
