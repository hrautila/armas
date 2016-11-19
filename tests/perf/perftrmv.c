
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "unit/testing.h"

double unitcent(int i, int j) {
  return 100.0*unitrand(i, j);
}

int main(int argc, char **argv)
{

  int verbose = 0;
  int count = 5;
  double rt, min, max, avg;
  armas_conf_t conf;
  armas_x_dense_t X, Y, Y0, Y1, A, At;

  int ok, opt, i;
  int N = 1701;
  int nproc = 1;
  int bsize = 0;
  int algo = 'B';

  while ((opt = getopt(argc, argv, "a:c:v")) != -1) {
    switch (opt) {
    case 'a':
      algo = *optarg;
      break;
    case 'v':
      verbose = 1;
      break;
    case 'c':
      count = atoi(optarg);
      break;
    default:
      fprintf(stderr, "usage: perftrmv [-a algo -c count -v] [size]\n");
      exit(1);
    }
  }
    
  if (optind < argc)
    N = atoi(argv[optind]);

  long seed = (long)time(0);
  srand48(seed);

  conf.mb = bsize == 0 ? 64  : bsize;
  conf.nb = bsize == 0 ? 96  : bsize;
  conf.kb = bsize == 0 ? 160 : bsize ;
  conf.maxproc = 1;
  switch (algo) {
  case 'N':
  case 'n':
    conf.optflags = ARMAS_ONAIVE;
    break;
  case 'R':
  case 'r':
  default:
    conf.optflags = ARMAS_ORECURSIVE;
    break;
  }    

  armas_x_init(&Y, N, 1);
  armas_x_init(&X, N, 1);
  armas_x_init(&A, N, N);
  
  armas_x_set_values(&X, unitcent, ARMAS_NULL);
  armas_x_set_values(&A, unitrand, ARMAS_SYMM);
  armas_x_mcopy(&Y, &X);

  // C = A*B
  min = max = avg = 0.0;
  for (i = 0; i < count; i++) {
    flush();
    rt = time_msec();

    armas_x_mvmult_trm(&X, &A, 1.0, ARMAS_LOWER, &conf);
    
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

    armas_x_mcopy(&X, &Y);
  }

  int64_t nops = (int64_t)N*N;
  printf("N: %4d, %8.4f, %8.4f, %8.4f Gflops\n",
	 N, gflops(max, nops), gflops(avg, nops), gflops(min, nops));
}

// Local Variables:
// indent-tabs-mode: nil
// End:
