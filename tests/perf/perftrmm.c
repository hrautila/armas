
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

//#include "armas.h"
#include <armas/dmatrix.h>
#include "helper.h"

int main(int argc, char **argv)
{

  int ok, opt, i;
  int count = 5;
  int nproc = 2;
  int trans = 0;
  int right = 0;
  int lower = 0;
  int algo = 'B';
  int flags = 0;
  int N = 600;
  int verbose = 0;
  double rt, min, max, avg;
  armas_conf_t conf;
  armas_d_dense_t C, A, B;

  while ((opt = getopt(argc, argv, "vc:P:a:s:t:T:")) != -1) {
    switch (opt) {
    case 's':
      right = *optarg == 'R' || *optarg == 'r';
      break;
    case 't':
      lower = *optarg == 'L' || *optarg == 'l';
      break;
    case 'T':
      trans = *optarg == 'T' || *optarg == 't';
      break;
    case 'a':
      algo = *optarg;
      break;
    case 'v':
      verbose = 1;
      break;
    case 'c':
      count = atoi(optarg);
      break;
    case 'P':
      nproc = atoi(optarg);
      break;
    default: /* ? */
      fprintf(stderr, "Usage: time_symm [-v] [-c numtest] [-P nproc] size");
      break;
    }
  }

  if (optind < argc)
    N = atoi(argv[optind]);

  flags |= right ? ARMAS_RIGHT : ARMAS_LEFT;
  flags |= lower ? ARMAS_LOWER : ARMAS_UPPER;
  if (trans)
    flags |= ARMAS_TRANSA;

  //armas_init();
  conf = *armas_conf_default();
  /*conf.mb = 64; conf.nb = 128; conf.kb = 160;
    conf.maxproc = nproc; */
  if (algo == 'N' || algo == 'n') {
    conf.optflags |= ARMAS_ONAIVE;
  } else if (algo == 'R' || algo == 'r') {
    conf.optflags |= ARMAS_ORECURSIVE;
  }

  if (verbose) {
    printf(".mb=%d, .nb=%d, .kb=%d, .wb=%d\n", conf.mb, conf.nb, conf.kb,conf.wb);
    printf(".maxproc=%d\n", conf.maxproc);
  }
  armas_d_init(&A, N, N);
  armas_d_init(&B, N, N);
  
  armas_d_set_values(&A, one, flags);
  armas_d_set_values(&B, one, ARMAS_NULL);

  // C = A*B
  min = max = avg = 0.0;
  for (i = 0; i < count; i++) {
    flush();
    rt = time_msec();

    armas_d_mult_trm(&B, &A, 1.0, flags, &conf);
    
    rt = time_msec() - rt;

    armas_d_set_values(&B, one, ARMAS_NULL);
    
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
  // forward/backward multiplication N^2 ops, and for N columns
  int64_t nops = (int64_t)N*N*N;
  printf("N: %4d, %8.4f, %8.4f, %8.4f Gflops\n",
	 N, gflops(max, nops), gflops(avg, nops), gflops(min, nops));
}

// Local Variables:
// indent-tabs-mode: nil
// End:
