
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "evd"

int test_1(int N, int flags, int verbose)
{
    armas_d_dense_t A, A0, D, W, I, sD, V, T;
    armas_conf_t conf = *armas_conf_default();
    double n0, n1;
    int err, ok;
    char *uplo = flags & ARMAS_LOWER ? "L" : "U";
    
    armas_d_init(&A, N, N);
    armas_d_init(&A0, N, N);
    armas_d_init(&V, N, N);
    armas_d_init(&I, N, N);
    armas_d_init(&D, N, 1);
    armas_d_init(&W, N*N, 1);

    armas_d_set_values(&A, unitrand, ARMAS_SYMM);
    armas_d_mcopy(&A0, &A);

    err = armas_d_eigen_sym(&D, &A, &W, flags|ARMAS_WANTV, &conf);
    if (err) {
        printf("err = %d, %d\n", err, conf.error);
        return 0;
    }
    
    armas_d_diag(&sD, &I, 0);
    armas_d_mult(&I, &A, &A, 1.0, 0.0, ARMAS_TRANSB, &conf);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_d_mnorm(&I, ARMAS_NORM_ONE, &conf);

    armas_d_mcopy(&V, &A);
    armas_d_mult_diag(&V, &D, ARMAS_RIGHT, &conf);
    armas_d_mult(&A0, &V, &A, -1.0, 1.0, ARMAS_TRANSB, &conf);

    if (N < 10 && verbose > 2) {
        printf("D:\n"); armas_d_printf(stdout, "%6.3f", armas_d_col_as_row(&T, &D));
        printf("I - V.T*V:\n"); armas_d_printf(stdout, "%6.3f", &I);
        printf("A - V*D*V.T:\n"); armas_d_printf(stdout, "%6.3f", &A0);
    }        
    n1 = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    ok = isOK(n1, 10*N);

    printf("%s [N=%d, uplo='%s']: A == V*D*V.T\n", PASS(ok), N, uplo);
    if (verbose > 0) {
        printf("  ||A - V*D*V.T||_1: %e [%ld]\n", n1, (long)(n1/DBL_EPSILON));
        printf("  ||I - V.T*V||_1  : %e [%ld]\n", n0, (long)(n0/DBL_EPSILON));
    }
    return ok;
}




main(int argc, char **argv)
{
    int opt;
    int M = 313;
    int N = 313;
    int ok = 0;
    int verbose = 0;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [N]\n", NAME);
            exit(1);
        }
    }
    
    if (optind < argc-1) {
        N = atoi(argv[optind]);
        M = atoi(argv[optind+1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;

    if (! test_1(N, ARMAS_LOWER, verbose))
        fails++;

    if (! test_1(N, ARMAS_UPPER, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
