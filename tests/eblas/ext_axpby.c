
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

/*
 *   A - B = d, where , d > 0, A >= d/ulp, B >= d/ulp
 *              and ulp is smallest number with 1 + ulp != 1
 *
 *   A = a*x, B = A - d = b*y - d
 *
 *   a = (1 + ulp)/sqrt(ulp), b = -1/sqrt(ulp)
 *   x = d/sqrt(ulp),         y = d/sqrt(ulp)
 *
 *   a*x + b*y = (d + d*ulp)/ulp - d/ulp = d
 */
int test_ext(int N, int verbose, DTYPE d, armas_conf_t *cf)
{
    armas_d_dense_t X, Y;
    DTYPE n0, alpha, beta;
    // EPS defined in lapack style as largest number with 1+eps == 1
    DTYPE ulp = 2.0*EPS;
    int ok, fails = 0;

    armas_d_init(&Y, N, 1);
    armas_d_init(&X, N, 1);
    for (int i = 0; i < N; ++i) {
        armas_d_set_at(&X, i, d/SQRT(ulp));
        armas_d_set_at(&Y, i, d/SQRT(ulp));
    }
    alpha = - ONE/SQRT(ulp);
    beta = (ONE + ulp)/SQRT(ulp);
    // expect result:
    // - extended precision: Y = d^T
    n0 = ZERO;
    armas_x_ext_axpby(beta, &Y, alpha, &X, cf);
    if (verbose > 2) {
        armas_x_dense_t r;
        MAT_PRINT("Y", armas_x_col_as_row(&r, &Y));
    }
    armas_x_madd(&Y, -d, 0, cf);
    n0 = armas_x_nrm2(&Y, cf);
    ok = n0 == ZERO || isOK(n0, N);
    fails += 1 - ok;
    printf("%6s: b*y + a*x == d\n", PASS(ok));
    if (verbose) {
        printf("   || rel error || = %.6e\n", n0);
    }
    if (verbose > 1) {
        for (int i = 0; i < N; ++i) {
            armas_d_set_at(&Y, i, d/SQRT(ulp));
        }
        armas_x_axpby(beta, &Y, alpha, &X, cf);
        if (verbose > 2) {
            armas_x_dense_t r;
            MAT_PRINT("Y", armas_x_col_as_row(&r, &Y));
        }
        DTYPE e0 = armas_x_get_at(&Y, 0);
        armas_x_madd(&Y, -d, 0, cf);
        n0 = armas_x_nrm2(&Y, cf);
        printf("   || normal precision || = %.6e [y0 = %e]\n", n0, e0);
    }
    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;
    int verbose = 0, opt, fails = 0;
    int N = 911;
    DTYPE diff = 19e4;

    cf = *armas_conf_default();
    env = armas_getenv();
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

    if (optind < argc) {
        N = atoi(argv[optind]);
    }
    fails += test_ext(N, verbose, diff, &cf);
    exit(fails);
}
