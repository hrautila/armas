
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"
//#include <armas/dmatrix.h>
//#include "helper.h"

int main(int argc, char **argv) {

  armas_conf_t conf;
  armas_x_dense_t X, Y, Y0, A, At;
  DTYPE nrm_y, nrm_z;
  
  int ok, opt;
  int verbose = 1;
  int N = 1307;
  int M = 1025;

  while ((opt = getopt(argc, argv, "v")) != -1) {
    switch (opt) {
    case 'v':
      verbose += 1;
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

  armas_x_init(&Y, M, 1);
  armas_x_init(&Y0, M, 1);
  armas_x_init(&X, N, 1);
  armas_x_init(&A, M, N);
  armas_x_init(&At, N, M);
  
  armas_x_set_values(&Y, zero, ARMAS_NULL);
  armas_x_set_values(&Y0, zero, ARMAS_NULL);
  armas_x_set_values(&X, unitrand, ARMAS_NULL);
  armas_x_set_values(&A, unitrand, ARMAS_NULL);
  armas_x_transpose(&At, &A);

  // Y = A*X
  armas_x_mvmult(&Y, &A, &X, 1.0, 0.0, 0, &conf);
  nrm_y = armas_x_nrm2(&Y, &conf);
  // Y = Y - A*X
  armas_x_mvmult(&Y, &At, &X, -1.0, 1.0, ARMAS_TRANS, &conf);
  if (N < 10 && verbose > 1) {
    printf("Y\n"); armas_x_printf(stdout, "%5.2f", &Y);
  }
  nrm_z = armas_x_nrm2(&Y, &conf);
  ok = nrm_z == 0.0 || isOK(nrm_z/nrm_y, N) ? 1 : 0;
  printf("%6s : gemv(A, X) == gemv(A.T, X, T)\n", PASS(ok));
  if (verbose > 0) {
    printf("   || rel error || : %e, [%d]\n", nrm_z/nrm_y, ndigits(nrm_z/nrm_y));
  }
  exit(1 - ok);

}

// Local Variables:
// indent-tabs-mode: nil
// End:
