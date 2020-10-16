
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_ext_expected(int M, int N, int verbose, armas_conf_t *cf)
{
    int fails = 0;

    armas_dense_t X, Y, Y0, A, At;
    DTYPE n0, n1;
    int ok;

    armas_init(&Y, M, 1);
    armas_init(&Y0, M, 1);
    armas_init(&X, N, 1);
    armas_init(&A, M, N);
    armas_init(&At, N, M);

    armas_set_values(&Y, zero, ARMAS_NULL);
    armas_set_values(&Y0, zero, ARMAS_NULL);
    armas_set_values(&X, one, ARMAS_NULL);
    make_ext_matrix_data(&A, 1.0, &Y0, ARMAS_LEFT);
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    if (verbose > 2) {
        MAT_PRINT("A", &A);
        MAT_PRINT("Y0", &Y0);
    }

    // Y = A*X
    armas_ext_mvmult(ZERO, &Y, ONE, &A, &X, 0, cf);
    n0 = rel_error(&n1, &Y, &Y0, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    printf("%6s : expected == ext_gemv(A, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 1) {
        armas_mvmult(ZERO, &Y, ONE, &A, &X, 0, cf);
        n0 = rel_error(&n1, &Y, &Y0, ARMAS_NORM_INF, 0, cf);
        printf("   || rel error || : %e, [%d] for standard precision\n", n0, ndigits(n0));
    }

    // Y = A^T*X
    armas_ext_mvmult(ZERO, &Y, ONE, &At, &X, ARMAS_TRANS, cf);
    n0 = rel_error(&n1, &Y, &Y0, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += 1 - ok;
    printf("%6s : expected == ext_gemv(A^T, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    if (verbose > 1) {
        armas_mvmult(ZERO, &Y, ONE, &At, &X, ARMAS_TRANS, cf);
        n0 = rel_error(&n1, &Y, &Y0, ARMAS_NORM_INF, 0, cf);
        printf("   || rel error || : %e, [%d] for standard precision\n", n0, ndigits(n0));
    }
    armas_release(&Y);
    armas_release(&Y0);
    armas_release(&X);
    armas_release(&A);
    armas_release(&At);

    return fails;
}

// Compute: A*x - A^T*x
int test_ext_trans(int M, int N, int verbose, armas_conf_t *cf)
{
    int fails = 0;

    armas_dense_t X, Y, Y0, A, At, t;
    DTYPE n0;
    int ok;

    armas_init(&Y, M, 1);
    armas_init(&Y0, M, 1);
    armas_init(&X, N, 1);
    armas_init(&A, M, N);
    armas_init(&At, N, M);

    armas_set_values(&Y, zero, ARMAS_NULL);
    armas_set_values(&Y0, zero, ARMAS_NULL);
    armas_set_values(&X, almost_one, ARMAS_NULL);
    make_ext_matrix_data(&A, 1.0, &Y0, ARMAS_LEFT);
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    if (verbose > 2) {
        MAT_PRINT("A", &A);
    }

    // Y = A*x; Y = Y - A^T*x
    armas_ext_mvmult(ZERO, &Y, ONE, &A, &X, 0, cf);
    armas_ext_mvmult(ONE, &Y, -ONE, &At, &X, ARMAS_TRANS, cf);
    n0 = armas_nrm2(&Y, cf);
    if (verbose > 1) {
        MAT_PRINT("Y", armas_col_as_row(&t, &Y));
    }
    ok = n0 == 0.0 || isOK(n0, N);
    printf("%6s : ext_gemv(A, X) == ext_gemv(A^T, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    armas_release(&Y);
    armas_release(&Y0);
    armas_release(&X);
    armas_release(&A);
    armas_release(&At);
    return fails;
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
            fprintf(stderr, "usage: gemv -n -v -r K [size]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    fails += test_ext_expected(M, N, verbose, &cf);
    fails += test_ext_trans(M, N, verbose, &cf);

    exit(fails);
}
