
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"
//#include <armas/dmatrix.h>
//#include "helper.h"

main(int argc, char **argv)
{

    armas_conf_t conf;
    armas_d_dense_t X, Y, X0, A, A0;

    int ok, opt;
    int M = 1013, N = 911;
    int flags = 0;
    int nproc = 1;
    int lower = 0;
    int trans = 0;
    int bsize = 0;
    int psize = 10;
    int algo = 'N';

    while ((opt = getopt(argc, argv, "P:")) != -1) {
        switch (opt) {
        case 'P':
            nproc = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: ger [-P N] [M N]\n");
            exit(1);
        }
    }

    if (optind < argc - 1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    conf.mb = bsize == 0 ? 64 : bsize;
    conf.nb = bsize == 0 ? 96 : bsize;
    conf.kb = bsize == 0 ? 160 : bsize;
    conf.maxproc = 1;

    armas_d_init(&Y, N, 1);
    armas_d_init(&X, M, 1);
    armas_d_init(&A, M, N);
    armas_d_init(&A0, M, N);

    armas_d_set_values(&X, unitrand, ARMAS_NULL);
    armas_d_set_values(&Y, unitrand, ARMAS_NULL);
    armas_d_set_values(&A, zero, flags);
    armas_d_mcopy(&A0, &A);

    armas_d_mvupdate(&A, 1.0, &X, &Y, &conf);
    armas_d_mvupdate(&A, -1.0, &X, &Y, &conf);
    ok = armas_d_allclose(&A, &A0);
    printf("%6s : A = A + X*Y - X*Y\n", ok ? "OK" : "FAILED");
    exit(1 - ok);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
