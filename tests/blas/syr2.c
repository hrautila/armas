
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_d_dense_t X, Y, A, A0;
    int ok;
    DTYPE n0, n1;
    const char *uplo = (flags & ARMAS_UPPER) ? "upper" : "lower";

    armas_d_init(&Y, N, 1);
    armas_d_init(&A, N, N);
    armas_d_init(&A0, N, N);

    armas_d_set_values(&Y, unitrand, 0);
    armas_d_set_values(&X, unitrand, 0);
    armas_d_set_values(&A, unitrand, flags);
    armas_d_mcopy(&A0, &A, 0, cf);

    printf("** symmetric rank-2 update: %s\n", uplo);

    // compute 2 ways; sym update/update + make upper
    armas_d_mvupdate2_sym(2.0, &A, 2.0, &X, &Y, flags, cf);

    armas_d_mvupdate(2.0, &A0, 2.0, &X, &Y, cf);
    armas_d_mvupdate(1.0, &A0, 2.0, &Y, &X, cf);
    armas_d_make_trm(&A0, flags);

    n0 = rel_error(&n1, &A, &A0, ARMAS_NORM_ONE, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : symupd(%s(A), x, y) == %s(update(A, x, y))\n",
           PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    return 1 - ok;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;
    int opt;
    int fails = 0;
    int N = 911;
    int verbose = 0;
    int all = 1;
    int upper = 0;
    int lower = 0;

    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vrUL")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'L':
            lower = 1;
            all = 0;
            break;
        case 'U':
            upper = 1;
            all = 0;
            break;
        default:
            fprintf(stderr, "usage: xsyrk [-v] [size]\n");
            exit(1);
        }
    }

    if (optind < argc)
        N = atoi(argv[optind]);

    if (all) {
        fails += test_std(N, verbose, ARMAS_LOWER, &cf);
        fails += test_std(N, verbose, ARMAS_UPPER, &cf);
    } else {
        if (lower)
            fails += test_std(N, verbose, ARMAS_LOWER, &cf);
        if (upper)
            fails += test_std(N, verbose, ARMAS_UPPER, &cf);
    }

    exit(fails);
}
