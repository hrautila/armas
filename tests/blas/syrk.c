
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_x_dense_t C, C0, A, At;
    DTYPE n0, n1;
    int ok;
    int fails = 0;
    DTYPE alpha = 2.0;
    const char *uplo;

    armas_x_init(&C, N, N);
    armas_x_init(&C0, N, N);
    armas_x_init(&A, N, N / 2);
    armas_x_init(&At, N / 2, N);

    armas_x_set_values(&A, zeromean, ARMAS_NULL);
    armas_x_mcopy(&At, &A, ARMAS_TRANS, cf);

    printf("** symmetric rank-k update: %s\n", (flags & ARMAS_UPPER) ? "upper" : "lower");
    uplo = (flags & ARMAS_UPPER) ? "U" : "L";

    // 1. C = upper(C) + A*A.T;
    armas_x_set_values(&C, one, ARMAS_SYMM);
    armas_x_mcopy(&C0, &C, 0, cf);

    armas_x_make_trm(&C, flags);
    armas_x_update_sym(0.0, &C, alpha, &A, flags, cf);

    armas_x_mult(0.0, &C0, alpha, &A, &A, ARMAS_TRANSB, cf);
    armas_x_make_trm(&C0, flags);
    if (verbose > 1) {
        MAT_PRINT("syrk(C)", &C);
        MAT_PRINT("C0", &C0);
    }

    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, 0, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

    printf("%6s: syrk(C, A, %c|N) == Tri%c(gemm(C, A, A.T))\n", PASS(ok), *uplo, *uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // 2. C = upper(C) + A.T*A
    armas_x_set_values(&C, one, ARMAS_SYMM);
    armas_x_mcopy(&C0, &C, 0, cf);

    armas_x_make_trm(&C, flags);
    armas_x_update_sym(0.0, &C, alpha, &At, flags | ARMAS_TRANSA, cf);

    armas_x_mult(0.0, &C0, alpha, &At, &At, ARMAS_TRANSA, cf);
    armas_x_make_trm(&C0, flags);

    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, 0, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;

    printf("%6s: syrk(C, A, %c|T) == Tri%c(gemm(C, A.T, A))\n", PASS(ok), *uplo, *uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t conf;

    int opt;
    int N = 255;
    int verbose = 1;
    int fails = 0;
    int lower = 0;
    int upper = 0;
    int all = 1;

    while ((opt = getopt(argc, argv, "vUL")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
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

    conf = *armas_conf_default();

    if (all) {
        fails += test_std(N, verbose, ARMAS_LOWER, &conf);
        fails += test_std(N, verbose, ARMAS_UPPER, &conf);
    } else {
        if (lower)
            fails += test_std(N, verbose, ARMAS_LOWER, &conf);
        if (upper)
            fails += test_std(N, verbose, ARMAS_UPPER, &conf);
    }
    return fails;
}
