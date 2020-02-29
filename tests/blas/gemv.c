
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int M, int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_x_dense_t X, Y, Y0, A, At;
    DTYPE n0, n1;
    int ok;
    const char *test = (flags & ARMAS_TRANS)
        ? "gemv(A^T, x) == A^T*x" : "gemv(A, x) == A*x";

    armas_x_init(&Y, M, 1);
    armas_x_init(&Y0, M, 1);
    armas_x_init(&X, N, 1);
    armas_x_init(&A, M, N);
    armas_x_init(&At, N, M);

    armas_x_set_values(&Y, unitrand, ARMAS_NULL);
    armas_x_set_values(&X, unitrand, ARMAS_NULL);
    armas_x_set_values(&A, unitrand, ARMAS_NULL);
    armas_x_mcopy(&At, &A, ARMAS_TRANS, cf);
    armas_x_mcopy(&Y0, &Y, 0, cf);

    // Y = A*X
    if (flags & ARMAS_TRANS) {
        armas_x_mvmult(2.0, &Y, ONE, &At, &X, ARMAS_TRANS, cf);
        for (int i = 0; i < M; i++) {
            armas_x_dense_t a0;
            armas_x_column(&a0, &At, i);
            DTYPE yk = 2.0 * armas_x_get_at_unsafe(&Y0, i);
            armas_x_adot(&yk, ONE, &a0, &X, cf);
            armas_x_set_at_unsafe(&Y0, i, yk);
        }
    } else {
        armas_x_mvmult(2.0, &Y, ONE, &A, &X, 0, cf);
        for (int i = 0; i < M; i++) {
            armas_x_dense_t a0;
            armas_x_row(&a0, &A, i);
            DTYPE yk = 2.0 * armas_x_get_at_unsafe(&Y0, i);
            armas_x_adot(&yk, ONE, &a0, &X, cf);
            armas_x_set_at_unsafe(&Y0, i, yk);
        }
    }
    // armas_x_mvmult(1.0, &Y, -1.0, &At, &X, ARMAS_TRANS, cf);
    if (N < 10 && verbose > 1) {
        printf("Y\n"); armas_x_printf(stdout, "%5.2f", &Y);
    }
    n0 = rel_error(&n1, &Y, &Y0, ARMAS_NORM_TWO, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : %s\n", PASS(ok), test);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_x_release(&Y);
    armas_x_release(&Y0);
    armas_x_release(&X);
    armas_x_release(&A);
    armas_x_release(&At);
    return 1 - ok;
}

int main(int argc, char **argv)
{

    armas_conf_t cf;
    armas_env_t *env;

    int opt, fails = 0;
    int verbose = 1;
    int N = 1307;
    int M = 1025;

    cf = *armas_conf_default();
    env = armas_getenv();

    while ((opt = getopt(argc, argv, "vnr:")) != -1) {
        switch (opt) {
        case 'n':
            cf.optflags |= ARMAS_ONAIVE;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: gemv [-nv -r num M N]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    fails += test_std(M, N, verbose, 0, &cf);
    fails += test_std(M, N, verbose, ARMAS_TRANS, &cf);

    exit(fails);
}
