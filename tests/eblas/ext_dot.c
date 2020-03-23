
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_ext(int N, int verbose, armas_conf_t *cf)
{
    armas_d_dense_t X, Y;
    DTYPE n0, n1, e1;
    int ok, fails = 0;

    armas_d_init(&Y, N, 1);
    armas_d_init(&X, N, 1);
    for (int i = 0; i < N; ++i) {
        armas_d_set_at(&X, i, EPS);
        armas_d_set_at(&Y, i, 2.0);
    }
    armas_d_set_at(&X, 0, N*1000.0);
    armas_d_set_at(&X, N-1, -N*1000.0);
    // expect result:
    // - extended precision: (2N-4)*epsilon
    // - standard precision: < (2N-4)*epsilon due to cancelation
    e1 = (2.0*N - 4.0)*EPS;
    n0 = ZERO;
    armas_x_adot(&n0, ONE, &X, &Y, cf);
    n1 = ZERO;
    armas_x_ext_adot(&n1, ONE, &X, &Y, cf);
    if (verbose > 1) {
        printf("computed: dot: %.16e, ext_dot: %.16e\n", n0, n1);
    }
    ok = n1 - e1 == ZERO || isOK(n1-e1, N);
    fails += 1 - ok;
    printf("%6s: ext(x^T*y) == (2N-4)*e\n", PASS(ok));
    ok = n1 > n0;
    printf("%6s: ext(x^T*y) >  std(x^T*y)\n", PASS(ok));
    if (verbose) {
        printf("   ||ext-std|| = %.6e [%d < %d]\n",
                n1 - n0, (int)((n1 - n0)/EPS/2.0), N);
    }
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
    fails += test_ext(N, verbose, &cf);
    exit(fails);
}
