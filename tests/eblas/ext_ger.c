
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

/*
 *   A - B = d, where d > 0, A >= d/ulp, B >= d/ulp
 *              and ulp is smallest number with 1 + ulp != 1
 *
 *   alpha*A + beta*x*y^T  =>  a_ij = alpha*a_ij + beta*x_i*y_j
 *
 *   a_ij = d/sqrt(ulp)     alpha = -1/sqrt(ulp)
 *    y_j = sqrt(d/ulp)     beta  = sqrt(d/ulp)
 *    x_i = 1 + k*ulp       k     = i + 1
 *
 *  => a_ij = -d/ulp + (1 + k*ulp)(d/ulp)  = -d/ulp + d/ulp + k*d  = k*d
 *
 */
int test_ext(int M, int N, int verbose, DTYPE d, armas_conf_t *cf)
{
    armas_d_dense_t X, Y, A, A0;
    DTYPE n0, n1, alpha, beta;
    // EPS defined in lapack style as largest number with 1+eps == 1
    DTYPE ulp = 2.0*EPS;
    int ok, fails = 0;

    armas_d_init(&A, M, N);
    armas_d_init(&A0, M, N);
    armas_d_init(&X, M, 1);
    armas_d_init(&Y, N, 1);

    for (int i = 0; i < M; i++) {
        armas_d_set_at(&X, i, ONE + ulp);
        for (int j = 0; j < N; j++) {
            armas_d_set(&A0, i, j, d);
            armas_d_set(&A,  i, j, d/SQRT(ulp));
            armas_d_set_at(&Y, j, SQRT(d/ulp));
        }
    }
    alpha = - ONE/SQRT(ulp);
    beta = SQRT(d/ulp);

    armas_x_ext_mvupdate(alpha, &A, beta, &X, &Y, cf);
    if (verbose > 2) {
        MAT_PRINT("A", &A);
    }
    n0 = rel_error(&n1, &A, &A0, ARMAS_NORM_INF, 0, cf);
    ok = n0 == ZERO || isOK(n0, N);
    fails += 1 - ok;
    printf("%6s: beta*A + alpha*x*y^T == expected\n", PASS(ok));
    if (verbose) {
        printf("   || rel error || = %.6e\n", n0);
    }
    if (verbose > 1) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                armas_d_set(&A,  i, j, d/SQRT(ulp));
            }
        }
        armas_x_mvupdate(alpha, &A, beta, &X, &Y, cf);
        if (verbose > 2) {
            MAT_PRINT("A", &A);
        }
        n0 = rel_error(&n1, &A, &A0, ARMAS_NORM_INF, 0, cf);
        printf("   || normal precision || = %.6e \n", n0);
    }
    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    int verbose = 0, opt, fails = 0;
    int M = 955;
    int N = 911;
    DTYPE diff = 19e4;

    cf = *armas_conf_default();
    while ((opt = getopt(argc, argv, "vD:")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'D':
            diff = atof(optarg);
            break;
        default:
            fprintf(stderr, "usage: axpby [-v] [N]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    fails += test_ext(M, N, verbose, diff, &cf);
    exit(fails);
}
