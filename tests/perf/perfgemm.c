
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t __Matrix ;
typedef float __Dtype;

#define PREC "single"

#define matrix_init       armas_s_init
#define matrix_set_values armas_s_set_values
#define matrix_mult       armas_s_mult
#define matrix_transpose  armas_s_transpose
#define matrix_release    armas_s_release
#define matrix_printf     armas_s_printf
#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define PREC "double"

#define matrix_init       armas_d_init
#define matrix_set_values armas_d_set_values
#define matrix_mult       armas_d_mult
#define matrix_transpose  armas_d_transpose
#define matrix_release    armas_d_release
#define matrix_printf     armas_d_printf

#endif
#include "helper.h"

main(int argc, char **argv) {

  int ok, opt, i;
  int count = 5;
  int nproc = 0;
  int N = 600;
  int verbose = 0;
  double rt, min, max, avg;
  armas_conf_t conf;
  __Matrix C, A, B;

  armas_init();
  conf = *armas_conf_default();

  while ((opt = getopt(argc, argv, "vc:P:oE")) != -1) {
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
    case 'E':
      conf.optflags |= ARMAS_OEXTPREC; // extended precision
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

  if (nproc > 0)
    conf.maxproc = nproc;

  matrix_init(&C, N, N);
  matrix_init(&A, N, N);
  matrix_init(&B, N, N);
  
  matrix_set_values(&C, zero, ARMAS_NULL);
  matrix_set_values(&A, unitrand, ARMAS_NULL);
  matrix_set_values(&B, unitrand, ARMAS_NULL);

  // C = A*B
  min = max = avg = 0.0;
  for (i = 0; i < count; i++) {
    flush();
    rt = time_msec();

    matrix_mult(&C, &A, &B, 1.0, 0.0, 0, &conf);
    
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
