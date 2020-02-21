
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

static
int test_row_vector(int N, int verbose, int flags)
{
    armas_conf_t conf = *armas_conf_default();
    armas_x_dense_t Z, X, A, X0;
    int ok, fails = 0;
    double n0, n1;
    char uplo = flags & ARMAS_UPPER ? 'U' : 'L';

    armas_x_init(&Z, N + 2, N);
    armas_x_row(&X0, &Z, 0);
    armas_x_row(&X, &Z, 1);
    armas_x_submatrix(&A, &Z, 2, 0, N, N);

    armas_x_set_values(&X0, one, ARMAS_NULL);
    armas_x_set_values(&X, one, ARMAS_NULL);
    armas_x_set_values(&A, zeromean, flags);
    if (flags & ARMAS_UNIT) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }
    printf("** trsv (row vector): %s %s\n",
           (flags & ARMAS_UPPER) ? "upper" : "lower",
           (flags & ARMAS_UNIT) ? "unit-diagonal" : "");
    if (verbose > 1) {
        MAT_PRINT("Z", &Z);
    }

    armas_x_mvmult(0.0, &X, 1.0, &A, &X0, 0, &conf);
    if (verbose > 1) {
        MAT_PRINT("X = A*X0", &X);
    }
    armas_x_mvsolve_trm(&X, 1.0, &A, flags, &conf);
    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    if (verbose > 2) {
        MAT_PRINT("result", &X);
    }
    printf("%6s : row(X) = trsv(trmv(X, A, %c|N), A, %c|N)\n", PASS(ok), uplo,
           uplo);
    fails += 1 - ok;
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }

    armas_x_mvmult(0.0, &X, 1.0, &A, &X0, ARMAS_TRANS, &conf);
    armas_x_mvsolve_trm(&X, 1.0, &A, flags | ARMAS_TRANS, &conf);
    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : row(X) = trsv(trmv(X, A, %c|T), A, %c|T)\n", PASS(ok), uplo,
           uplo);
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }
    fails += 1 - ok;

    return fails;
}

static
int test_col_vector(int N, int verbose, int flags)
{
    armas_conf_t conf = *armas_conf_default();
    armas_x_dense_t Z, X, A, X0;
    int ok, fails = 0;
    double n0, n1;
    char uplo = flags & ARMAS_UPPER ? 'U' : 'L';

    armas_x_init(&Z, N, N + 2);
    armas_x_column(&X0, &Z, 0);
    armas_x_column(&X, &Z, 1);
    armas_x_submatrix(&A, &Z, 0, 2, N, N);

    armas_x_set_values(&X0, one, ARMAS_NULL);
    armas_x_set_values(&X, one, ARMAS_NULL);
    armas_x_set_values(&A, zeromean, flags);
    if (flags & ARMAS_UNIT) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }

    printf("** trsv (column vector): %s %s\n",
           (flags & ARMAS_UPPER) ? "upper" : "lower",
           (flags & ARMAS_UNIT) ? "unit-diagonal" : "");

    armas_x_mvmult(0.0, &X, 1.0, &A, &X0, 0, &conf);
    armas_x_mvsolve_trm(&X, 1.0, &A, flags, &conf);
    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    if (verbose > 1) {
        MAT_PRINT("Z", &Z);
    }
    printf("%6s : col(X) = trsv(trmv(X, A, %c|N), A, %c|N)\n", 
        PASS(ok), uplo, uplo);
    fails += 1 - ok;
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }

    armas_x_mvmult(0.0, &X, 1.0, &A, &X0, ARMAS_TRANS, &conf);
    armas_x_mvsolve_trm(&X, 1.0, &A, flags | ARMAS_TRANS, &conf);
    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_TWO, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    if (verbose > 1) {
        MAT_PRINT("Z", &Z);
    }
    printf("%6s : col(X) = trsv(trmv(X, A, %c|T), A, %c|T)\n",
        PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("  || error ||: %e\n", n0);
    }
    fails += 1 - ok;

    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 77;
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
        fails += test_col_vector(N, verbose, ARMAS_UPPER);
        fails += test_col_vector(N, verbose, ARMAS_UPPER|ARMAS_UNIT);
        fails += test_row_vector(N, verbose, ARMAS_LOWER);
        fails += test_row_vector(N, verbose, ARMAS_LOWER|ARMAS_UNIT);
        fails += test_col_vector(N, verbose, ARMAS_LOWER);
        fails += test_col_vector(N, verbose, ARMAS_LOWER|ARMAS_UNIT);
    } else {
        if (upper) {
            fails += test_row_vector(N, verbose, ARMAS_UPPER|unit);
            fails += test_col_vector(N, verbose, ARMAS_UPPER|unit);
        }
        if (lower) {
            fails += test_row_vector(N, verbose, ARMAS_LOWER|unit);
            fails += test_col_vector(N, verbose, ARMAS_LOWER|unit);
        }
    }
    exit(fails);
}
