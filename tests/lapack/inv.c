
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "inverse"

#if FLOAT32
#define ERROR 1e-4
#else
#define ERROR 1e-12
#endif

int test_inverse(int N, int lb, int verbose)
{
    armas_dense_t A0, A1, W;
    DTYPE n0, n1;
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();

    armas_init(&A0, N, N);
    armas_set_values(&A0, unitrand, 0);

    armas_init(&A1, N, N);
    armas_init(&W, N, lb == 0 ? 1 : lb);
    armas_pivot_init(&P, N);

    env->lb = lb;
    armas_lufactor(&A0, &P, &conf);
    armas_mcopy(&A1, &A0, 0, &conf);

    env->lb = 0;
    armas_luinverse(&A1, &P, &conf);
    if (verbose > 1) {
        MAT_PRINT("unblk.A^-1", &A1);
    }

    env->lb = lb;
    armas_luinverse(&A0, &P, &conf);
    if (verbose > 1) {
        MAT_PRINT("blk.A^-1", &A0);
    }

    n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * ERROR);
    fails += 1 - ok;
    printf("%s: unblk.A^-1 == blk.A^-1\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }
    armas_release(&A0);
    armas_release(&A1);
    armas_release(&W);

    return fails;
}

int test_equal(int N, int lb, int verbose)
{
    armas_dense_t A0, A1, W;
    DTYPE n0, n1;
    const char *blk = (lb == 0) ? "unblk" : "  blk";
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();

    armas_init(&A0, N, N);
    armas_set_values(&A0, unitrand, 0);

    armas_init(&A1, N, N);
    armas_init(&W, N, lb == 0 ? 1 : lb);
    armas_pivot_init(&P, N);

    armas_mcopy(&A1, &A0, 0, &conf);
    if (verbose > 1) {
        MAT_PRINT("A", &A1);
    }

    env->lb = lb;
    armas_lufactor(&A1, &P, &conf);
    if (verbose > 1) {
        MAT_PRINT("LU(A)", &A1);
    }
    if (armas_luinverse(&A1, &P, &conf) < 0)
        printf("inverse.1 error: %d\n", conf.error);
    if (verbose > 1) {
        MAT_PRINT("A^-1", &A1);
    }

    armas_lufactor(&A1, &P, &conf);
    if (verbose > 1) {
        MAT_PRINT("LU(A^-1)", &A1);
    }
    if (armas_luinverse(&A1, &P, &conf) < 0)
        printf("inverse.2 error: %d\n", conf.error);
    if (verbose > 1) {
        MAT_PRINT("LU(A^-1)^-1", &A1);
    }

    n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * ERROR);
    fails += 1 - ok;
    printf("%s: %s.(A.-1).-1 == A\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&W);
    return fails;
}

int test_ident(int N, int lb, int verbose)
{
    armas_dense_t A0, A1, C, W, D;
    DTYPE n0;
    char *blk = (lb == 0) ? "unblk" : "  blk";
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();

    armas_init(&A0, N, N);
    armas_init(&A1, N, N);
    armas_init(&C, N, N);
    armas_init(&W, N, (lb == 0) ? 1 : lb);
    armas_pivot_init(&P, N);

    armas_set_values(&A0, unitrand, 0);
    armas_mcopy(&A1, &A0, 0, &conf);
    armas_diag(&D, &C, 0);

    env->lb = lb;
    armas_lufactor(&A1, &P, &conf);
    armas_luinverse(&A1, &P, &conf);

    armas_mult(ZERO, &C, ONE, &A1, &A0, 0, &conf);
    armas_madd(&D, -ONE, 0, &conf);
    n0 = armas_mnorm(&C, ARMAS_NORM_INF, &conf);
    ok = isFINE(n0, N * ERROR);
    fails += 1 - ok;
    printf("%s: %s.A.-1*A == I\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&W);
    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 515;
    int LB = 36;
    int verbose = 1;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [M N LB]\n", NAME);
            exit(1);
        }
    }

    if (optind < argc - 1) {
        N = atoi(argv[optind]);
        LB = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        LB = N > 32 ? 16 : 8;
    }

    int fails = 0;
    fails += test_inverse(N, LB, verbose);

    fails += test_equal(N, 0, verbose);
    fails += test_equal(N, LB, verbose);
    fails += test_ident(N, 0, verbose);
    fails += test_ident(N, LB, verbose);

    exit(fails);
}
