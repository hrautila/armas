
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-14
#endif

#define NAME "evd"

int test_1(int N, int flags, int verbose)
{
    __Matrix A, A0, D, W, I, sD, V, T;
    armas_conf_t conf = *armas_conf_default();
    __Dtype n0, n1, nrm_A;
    int err, ok;
    char *uplo = flags & ARMAS_LOWER ? "L" : "U";
    
    matrix_init(&A, N, N);
    matrix_init(&A0, N, N);
    matrix_init(&V, N, N);
    matrix_init(&I, N, N);
    matrix_init(&D, N, 1);
    matrix_init(&W, N*N, 1);

    matrix_set_values(&A, unitrand, ARMAS_SYMM);
    matrix_mcopy(&A0, &A);
    nrm_A = matrix_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    
    err = matrix_eigen_sym(&D, &A, &W, flags|ARMAS_WANTV, &conf);
    if (err) {
        printf("err = %d, %d\n", err, conf.error);
        return 0;
    }
    
    matrix_diag(&sD, &I, 0);
    matrix_mult(&I, &A, &A, 1.0, 0.0, ARMAS_TRANSB, &conf);
    matrix_madd(&sD, -1.0, ARMAS_ANY);
    n0 = matrix_mnorm(&I, ARMAS_NORM_ONE, &conf);

    matrix_mcopy(&V, &A);
    matrix_mult_diag(&V, &D, ARMAS_RIGHT, &conf);
    matrix_mult(&A0, &V, &A, -1.0, 1.0, ARMAS_TRANSB, &conf);

    if (N < 10 && verbose > 2) {
        printf("D:\n"); matrix_printf(stdout, "%6.3f", matrix_col_as_row(&T, &D));
        printf("I - V.T*V:\n"); matrix_printf(stdout, "%6.3f", &I);
        printf("A - V*D*V.T:\n"); matrix_printf(stdout, "%6.3f", &A0);
    }        
    n1 = matrix_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    n1 /= nrm_A;
    ok = isOK(n1, N);

    printf("%s [N=%d, uplo='%s']: A == V*D*V.T\n", PASS(ok), N, uplo);
    if (verbose > 0) {
        printf("  ||A - V*D*V.T||_1: %e [%d]\n", n1, ndigits(n1));
        printf("  ||I - V.T*V||_1  : %e [%d]\n", n0, ndigits(n0));
    }
    return ok;
}




int main(int argc, char **argv)
{
    int opt;
    int N = 313;
    int verbose = 1;

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
    
    if (optind < argc) {
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
