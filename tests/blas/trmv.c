
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
    armas_d_dense_t X, Y, X0, A, At;

    int ok, opt;
    int N = 911;
    int nproc = 1;
    int fails = 0;

    while ((opt = getopt(argc, argv, "P:")) != -1) {
        switch (opt) {
        case 'P':
            nproc = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: trmv [size]\n");
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


    armas_d_init(&Y, N, 1);
    armas_d_init(&X0, N, 1);
    armas_d_init(&X, N, 1);
    armas_d_init(&A, N, N);
    armas_d_init(&At, N, N);

    armas_d_set_values(&X, unitrand, ARMAS_NULL);
    armas_d_set_values(&Y, zero, ARMAS_NULL);
    armas_d_mcopy(&X0, &X);

    armas_d_set_values(&A, unitrand, ARMAS_UPPER);
    armas_d_transpose(&At, &A);

    // trmv(A, X) == gemv(A, X) 
    armas_d_mvmult_trm(&X, 1.0, &A, ARMAS_UPPER, &conf);
    armas_d_mvmult(0.0, &Y, 1.0, &A, &X0, ARMAS_NULL, &conf);
    ok = armas_d_allclose(&Y, &X);
    printf("%6s : trmv(X, A, U|N) == gemv(upper(A), X)\n",
           ok ? "OK" : "FAILED");
    fails += 1 - ok;

    armas_d_mcopy(&X, &X0);

    // trmv(A.T, X) == gemv(A.T, X) 
    armas_d_mvmult_trm(&X, 1.0, &A, ARMAS_UPPER | ARMAS_TRANSA, &conf);
    armas_d_mvmult(0.0, &Y, 1.0, &A, &X0, ARMAS_TRANSA, &conf);
    ok = armas_d_allclose(&Y, &X);
    printf("%6s : trmv(X, A, U|T) == gemv(upper(A).T, X)\n",
           ok ? "OK" : "FAILED");
    fails += 1 - ok;

    armas_d_set_values(&A, zero, ARMAS_NULL);
    armas_d_set_values(&A, unitrand, ARMAS_LOWER);
    armas_d_transpose(&At, &A);

    armas_d_mcopy(&X, &X0);

    // trmv(A, X) == gemv(A, X) 
    armas_d_mvmult_trm(&X, 1.0, &A, ARMAS_LOWER, &conf);
    armas_d_mvmult(0.0, &Y, 1.0, &A, &X0, ARMAS_NULL, &conf);
    ok = armas_d_allclose(&Y, &X);
    printf("%6s : trmv(X, A, L|N) == gemv(lower(A), X)\n",
           ok ? "OK" : "FAILED");
    fails += 1 - ok;

    armas_d_mcopy(&X, &X0);

    // trmv(A.T, X) == gemv(A.T, X) 
    armas_d_mvmult_trm(&X, 1.0, &A, ARMAS_LOWER | ARMAS_TRANSA, &conf);
    armas_d_mvmult(0.0, &Y, 1.0, &A, &X0, ARMAS_TRANSA, &conf);
    ok = armas_d_allclose(&Y, &X);
    printf("%6s : trmv(X, A, L|T) == gemv(lower(A).T, X)\n",
           ok ? "OK" : "FAILED");
    fails += 1 - ok;

    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
