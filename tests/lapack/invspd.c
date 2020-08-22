
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#if FLOAT32
#define ERROR 1e-4
#else
#define ERROR 1e-8
#endif

#define NAME "invspd"

int test_ident(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, A2, C, W, D;
    DTYPE n0;
    const char *blk = (lb == 0) ? "unblk" : "  blk";
    const char uplo = (flags & ARMAS_LOWER) ? 'L' : 'U';
    int ok;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&A2, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&W, N, lb == 0 ? 1 : lb);

    // symmetric positive definite matrix A^T*A
    armas_x_set_values(&A0, unitrand, 0);
    armas_x_mult(ZERO, &A1, ONE, &A0, &A0, ARMAS_TRANSA, &conf);
    armas_x_make_trm(&A1, flags);
    armas_x_mcopy(&A0, &A1, 0, &conf);

    // identity; D = diag(C)
    armas_x_diag(&D, &C, 0);
    armas_x_set_values(&D, one, 0);

    env->lb = lb;
    armas_x_cholfactor(&A1, ARMAS_NOPIVOT, flags, &conf);
    armas_x_cholinverse(&A1, flags, &conf);

    // A2 = A1*I
    armas_x_mult_sym(ZERO, &A2, ONE, &A1, &C, flags | ARMAS_LEFT, &conf);
    armas_x_mult_sym(ZERO, &C, ONE, &A0, &A2, flags, &conf);

    // diag(C) -= 1.0
    armas_x_madd(&D, -ONE, 0, &conf);
    n0 = armas_x_mnorm(&C, ARMAS_NORM_INF, &conf);
    ok = isFINE(n0, N * ERROR);

    printf("%s: %c %s.A.-1*A == I\n", PASS(ok), uplo, blk);
    printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&A2);
    armas_x_release(&C);
    armas_x_release(&W);
    return 1 - ok;
}

int test_equal(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, W;
    DTYPE n0;
    const char *blk = (lb == 0) ? "unblk" : "  blk";
    const char uplo = (flags & ARMAS_LOWER) ? 'L' : 'U';
    int ok;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&W, N, lb == 0 ? 1 : lb);

    // symmetric positive definite matrix A^T*A
    armas_x_set_values(&A0, unitrand, 0);
    armas_x_mult(ZERO, &A1, ONE, &A0, &A0, ARMAS_TRANSA, &conf);
    armas_x_make_trm(&A1, flags);
    armas_x_mcopy(&A0, &A1, 0, &conf);

    env->lb = lb;
    armas_x_cholfactor(&A1, ARMAS_NOPIVOT, flags, &conf);
    armas_x_cholinverse(&A1, flags, &conf);

    armas_x_cholfactor(&A1, ARMAS_NOPIVOT, flags, &conf);
    armas_x_cholinverse(&A1, flags, &conf);

    n0 = rel_error(&n0, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * ERROR);

    printf("%s: %c %s.(A.-1).-1 == A\n", PASS(ok), uplo, blk);
    printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&W);
    return 1 - ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 177;
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
    }

    int fails = 0;
    fails += test_ident(N, 0, ARMAS_LOWER, verbose);
    fails += test_ident(N, LB, ARMAS_LOWER, verbose);
    fails += test_ident(N, 0, ARMAS_UPPER, verbose);
    fails += test_ident(N, LB, ARMAS_UPPER, verbose);
    fails += test_equal(N, 0, ARMAS_LOWER, verbose);
    fails += test_equal(N, LB, ARMAS_LOWER, verbose);
    fails += test_equal(N, 0, ARMAS_UPPER, verbose);
    fails += test_equal(N, LB, ARMAS_UPPER, verbose);
    exit(fails);
}
