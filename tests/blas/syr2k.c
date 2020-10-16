
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

static
int test_std(int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_dense_t C, C0, A, At, B, Bt;
    int ok, fails = 0;
    DTYPE n0, n1, alpha = 2.0;
    const char *uplo = (flags & ARMAS_UPPER) ? "U" : "L";

    armas_init(&C, N, N);
    armas_init(&C0, N, N);
    armas_init(&A, N, N / 2);
    armas_init(&At, N / 2, N);
    armas_init(&B, N, N / 2);
    armas_init(&Bt, N / 2, N);

    armas_set_values(&A, zeromean, 0);
    armas_set_values(&B, zeromean, 0);
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    armas_mcopy(&Bt, &B, ARMAS_TRANS, cf);

    printf("** symmetric rank-2k update: %s\n",
           (flags & ARMAS_UPPER) ? "upper" : "lower");
    // 1. C = C + A*B.T + B*A.T;
    armas_set_values(&C, one, ARMAS_SYMM);
    armas_mcopy(&C0, &C, 0, cf);

    armas_make_trm(&C, flags);
    armas_update2_sym(0.0, &C, alpha, &A, &B, flags, cf);

    armas_mult(0.0, &C0, alpha, &A, &Bt, 0, cf);
    armas_mult(1.0, &C0, alpha, &B, &At, 0, cf);
    armas_make_trm(&C0, flags);

    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: syr2k(C, A, %c|N) == Tri%c(C + A*B.T + B*A.T))\n",
           PASS(ok), *uplo, *uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // 2. C = C + B.T*A + A,T*B
    armas_set_values(&C, one, ARMAS_SYMM);
    armas_mcopy(&C0, &C, 0, cf);

    armas_make_trm(&C, flags);
    armas_update2_sym(0.0, &C, alpha, &At, &Bt, flags | ARMAS_TRANSA, cf);

    armas_mult(0.0, &C0, alpha, &Bt, &A, ARMAS_TRANSA | ARMAS_TRANSB, cf);
    armas_mult(1.0, &C0, alpha, &At, &B, ARMAS_TRANSA | ARMAS_TRANSB, cf);
    armas_make_trm(&C0, flags);

    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: syr2k(C, A, %c|T|N) == Tri%c(C + B.T*A + A.T*B))\n",
           PASS(ok), *uplo, *uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int main(int argc, char **argv)
{

    armas_conf_t cf;
    int opt;
    int N = 213;
    int verbose = 1;
    int fails = 0;
    int lower = 0;
    int upper = 0;
    int all = 1;
    armas_env_t *env = armas_getenv();
    cf = *armas_conf_default();

    while ((opt = getopt(argc, argv, "vULr:")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'r':
            env->lb = atoi(optarg);
            cf.optflags |= env->lb != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
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
