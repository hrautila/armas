
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <armas/dmatrix.h>
#include "helper.h"

main(int argc, char **argv) {

  int ok, opt, i;
  int count = 5;
  int nproc = 0;
  int N = 600;
  int verbose = 0;
  double rt, min, max, avg;
  armas_conf_t conf;
  armas_d_dense_t C, A, B;

  armas_init();
  conf = *armas_conf_default();

  while ((opt = getopt(argc, argv, "vc:P:o")) != -1) {
    switch (opt) {
    case 'v':
      verbose = 1;
      break;
    case 'c':
      count = atoi(optarg);
      break;
    case 'P':
      nproc = atoi(optarg);
      break;
    case 'o':
      conf.optflags |= ARMAS_BLAS_RECURSIVE;
      break;
    default: /* ? */
      fprintf(stderr, "Usage: time_gemm [-v] [-c numtest] [-P nproc] [size]");
      break;
    }
  }
  if (optind < argc)
    N = atoi(argv[optind]);

  long seed = (long)time(0);
  srand48(seed);

  //conf.mb = 96; conf.nb = 128; conf.kb = 160;
  if (nproc > 0)
    conf.maxproc = nproc;

  armas_d_init(&C, N, N);
  armas_d_init(&A, N, N);
  armas_d_init(&B, N, N);
  
  armas_d_set_values(&C, zero, ARMAS_NULL);
  armas_d_set_values(&A, unitrand, ARMAS_NULL);
  armas_d_set_values(&B, unitrand, ARMAS_NULL);

  // C = A*B
  min = max = avg = 0.0;
  for (i = 0; i < count; i++) {
    flush();
    rt = time_msec();

    armas_d_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
    
    rt = time_msec() - rt;

    if (i == 0) {
      min = max = avg = rt;
    } else {
      if (rt < min)
	min = rt;
      if (rt > max)
	max = rt;
      avg += (rt - avg)/(i+1);
    }
    if (verbose)
      printf("%2d: %.4f, %.4f, %.4f msec\n", i, min, avg, max);
  }
  int64_t nops = 2*(int64_t)N*N*N;
  printf("N: %4d, %8.4f, %8.4f, %8.4f Gflops\n",
	 N, gflops(max, nops), gflops(avg, nops), gflops(min, nops));
}

// Local Variables:
// indent-tabs-mode: nil
// End:
