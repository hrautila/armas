
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int N, int verbose, armas_conf_t *cf)
{
    armas_dense_t X, Y;
    int ok, fails = 0;
    DTYPE n0, n1;

    armas_init(&Y, N, 1);
    armas_init(&X, N, 1);

    armas_set_values(&X, one, ARMAS_NULL);
    armas_set_values(&Y, unitrand, ARMAS_NULL);

    n0 = armas_dot(&X, &X, cf);
    n0 = (double)N - n0;
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : 1*1^T - N == 0\n", PASS(ok));
    fails += 1 - ok;

    n0 = armas_dot(&X, &Y, cf);
    n1 = armas_asum(&Y, cf);

    n0 = n1 - n0;
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : 1*Y - asum(Y) == 0\n", PASS(ok));
    fails += 1 - ok;

    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;
    int verbose = 0, opt, fails = 0;
    int N = 911;

    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vr:n")) != -1) {
        switch (opt) {
        case 'n':
            cf.optflags |= ARMAS_ONAIVE;
            break;
        case 'r':
            env->blas1min = atoi(optarg);
            cf.optflags |= env->blas1min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'v':
            verbose++;
            break;
        default:
            fprintf(stderr, "usage: ger [-nrbv] [M N]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }
    fails += test_std(N, verbose, &cf);
    exit(fails);
}
