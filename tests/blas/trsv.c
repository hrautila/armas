
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

static
int test_row_vector(int N, int verbose, int flags)
{
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    armas_x_dense_t Z, X, A, X0;
    int ok, fails = 0;
    double n0, n1;
    char uplo = (flags & ARMAS_UPPER) ? 'U' : 'L';

    armas_x_init(&Z, N + 2, N);
    armas_x_row(&X0, &Z, 0);
    armas_x_row(&X, &Z, 1);
    armas_x_submatrix(&A, &Z, 2, 0, N, N);

    armas_x_set_values(&A, zeromean, flags);
    if (flags & ARMAS_UNIT) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, ONE);
        }
    }
    armas_x_set_values(&X0, one, ARMAS_NULL);
    printf("** trsv (row vector): %s %s %s\n",
           (flags & ARMAS_UPPER) ? "upper" : "lower",
           (flags & ARMAS_TRANS) ? "transpose" : "no-transpose",
           (flags & ARMAS_UNIT) ? "unit-diagonal" : "");

    env->blas2min = 0;
    armas_x_mvmult(ZERO, &X, TWO, &A, &X0, flags, &conf);
    armas_x_mvsolve_trm(&X, ONE/TWO, &A, flags, &conf);
    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;

    printf("%6s : row(X) = trsv(%c(A)*X, A)\n", PASS(ok), uplo);
    fails += 1 - ok;
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }
    armas_x_release(&Z);
    return fails;
}

static
int test_col_vector(int N, int verbose, int flags)
{
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    armas_x_dense_t Z, X, A, X0;
    int ok, fails = 0;
    double n0, n1;
    char uplo = (flags & ARMAS_UPPER) ? 'U' : 'L';

    armas_x_init(&Z, N, N + 2);
    armas_x_column(&X0, &Z, 0);
    armas_x_column(&X, &Z, 1);
    armas_x_submatrix(&A, &Z, 0, 2, N, N);

    armas_x_set_values(&A, zeromean, flags);
    if (flags & ARMAS_UNIT) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, ONE);
        }
    }
    armas_x_set_values(&X0, one, ARMAS_NULL);

    printf("** trsv (column vector): %s %s %s\n",
           (flags & ARMAS_UPPER) ? "upper" : "lower",
           (flags & ARMAS_TRANS) ? "transpose" : "no-transpose",
           (flags & ARMAS_UNIT) ? "unit-diagonal" : "");

    env->blas2min = 0;
    armas_x_mvmult(ZERO, &X, TWO, &A, &X0, flags, &conf);
    armas_x_mvsolve_trm(&X, ONE/TWO, &A, flags, &conf);
    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, 0, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    if (verbose > 1) {
        MAT_PRINT("Z", &Z);
    }
    printf("%6s : col(X) = trsv(%c(A)*X, A)\n", PASS(ok), uplo);
    fails += 1 - ok;
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }
    armas_x_release(&Z);
    return fails;
}

static
int test_blocked(int N, int verbose, int lb, int flags)
{
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    armas_x_dense_t Z, X1, A, X0;
    int ok, fails = 0;
    double n0, n1;

    armas_x_init(&Z, N, N + 2);
    armas_x_column(&X0, &Z, 0);
    armas_x_column(&X1, &Z, 1);
    armas_x_submatrix(&A, &Z, 0, 2, N, N);

    armas_x_set_values(&A, zeromean, flags);
    if (flags & ARMAS_UNIT) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, ONE);
        }
    }
    armas_x_set_values(&X0, one, ARMAS_NULL);
    armas_x_mvmult(ZERO, &X1, TWO, &A, &X0, flags, &conf);
    armas_x_mcopy(&X0, &X1, 0, &conf);

    printf("** trsv: %s %s %s\n",
           (flags & ARMAS_UPPER) ? "upper" : "lower",
           (flags & ARMAS_TRANS) ? "transpose" : "no-transpose",
           (flags & ARMAS_UNIT) ? "unit-diagonal" : "");

    env->blas2min = 0;
    armas_x_mvsolve_trm(&X0, ONE/TWO, &A, flags, &conf);
    env->blas2min = lb;
    armas_x_mvsolve_trm(&X1, ONE/TWO, &A, flags, &conf);
    n0 = rel_error(&n1, &X1, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    if (verbose > 1) {
        MAT_PRINT("Z", &Z);
    }
    printf("%6s : unblk.trsv(A, X) = blk.trsv(A, X)\n", PASS(ok));
    fails += 1 - ok;
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }
    armas_x_release(&Z);
    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 77;
    int lb = 16;
    int verbose = 0;
    int all = 1;
    int unit = 0;
    int lower = 0;
    int upper = 0;

    while ((opt = getopt(argc, argv, "vULu")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'u':
            unit = ARMAS_UNIT;
            break;
        case 'U':
            upper = 1;
            all = 0;
            break;
        case 'L':
            lower = 1;
            all = 0;
            break;
        default:
            fprintf(stderr, "usage: trsv [size]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;

    if (all) {
        fails += test_row_vector(N, verbose, ARMAS_UPPER);
        fails += test_row_vector(N, verbose, ARMAS_UPPER|ARMAS_UNIT);
        fails += test_row_vector(N, verbose, ARMAS_UPPER|ARMAS_TRANS);
        fails += test_row_vector(N, verbose, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT);
        fails += test_col_vector(N, verbose, ARMAS_UPPER);
        fails += test_col_vector(N, verbose, ARMAS_UPPER|ARMAS_UNIT);
        fails += test_col_vector(N, verbose, ARMAS_UPPER|ARMAS_TRANS);
        fails += test_col_vector(N, verbose, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT);
        fails += test_row_vector(N, verbose, ARMAS_LOWER);
        fails += test_row_vector(N, verbose, ARMAS_LOWER|ARMAS_UNIT);
        fails += test_row_vector(N, verbose, ARMAS_LOWER|ARMAS_TRANS);
        fails += test_row_vector(N, verbose, ARMAS_LOWER|ARMAS_TRANS|ARMAS_UNIT);
        fails += test_col_vector(N, verbose, ARMAS_LOWER);
        fails += test_col_vector(N, verbose, ARMAS_LOWER|ARMAS_UNIT);
        fails += test_col_vector(N, verbose, ARMAS_LOWER|ARMAS_TRANS);
        fails += test_col_vector(N, verbose, ARMAS_LOWER|ARMAS_TRANS|ARMAS_UNIT);
        fails += test_blocked(N, verbose, lb, ARMAS_UPPER);
        fails += test_blocked(N, verbose, lb, ARMAS_UPPER|ARMAS_TRANS);
        fails += test_blocked(N, verbose, lb, ARMAS_UPPER|ARMAS_UNIT);
        fails += test_blocked(N, verbose, lb, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT);
    } else {
        if (upper) {
            fails += test_row_vector(N, verbose, ARMAS_UPPER|unit);
            fails += test_col_vector(N, verbose, ARMAS_UPPER|ARMAS_TRANS|unit);
        }
        if (lower) {
            fails += test_row_vector(N, verbose, ARMAS_LOWER|unit);
            fails += test_col_vector(N, verbose, ARMAS_LOWER|unit);
        }
    }
    exit(fails);
}
