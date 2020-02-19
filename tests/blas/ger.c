
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int M, int N, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t X, Y, A, A0;
    int ok;
    DTYPE n0, n1;

    armas_x_init(&Y, N, 1);
    armas_x_init(&X, M, 1);
    armas_x_init(&A, M, N);
    armas_x_init(&A0, M, N);

    armas_x_set_values(&X, unitrand, 0);
    armas_x_set_values(&Y, unitrand, 0);
    armas_x_set_values(&A, unitrand, 0);
    armas_x_mcopy(&A0, &A, 0, cf);

    armas_x_mvupdate(1.0, &A, 1.0, &X, &Y, cf);
    armas_x_mvupdate(1.0, &A, -1.0, &X, &Y, cf);

    n0 = rel_error(&n1, &A, &A0, ARMAS_NORM_ONE, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : A == A + x*y^T - x*y^T\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    return (1 - ok);
}

int main(int argc, char **argv)
{

    armas_conf_t cf;
    armas_env_t *env;

    int opt;
    int verbose = 0;
    int fails = 0;
    int M = 1013, N = 911;

    cf = *armas_conf_default();
    env = armas_getenv();

    while ((opt = getopt(argc, argv, "vr:")) != -1) {
        switch (opt) {
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'v':
            verbose++;
            break;
        default:
            fprintf(stderr, "usage: ger [-nrbv] [M N]\n");
            exit(1);
        }
    }

    if (optind < argc - 1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    fails += test_std(M, N, verbose, &cf);
    exit(fails);
}
