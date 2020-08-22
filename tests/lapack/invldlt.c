/**
 * Non-pivoting LDL.T factoring
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "invldlt"

#if FLOAT32
#define ERROR 1e-4
#else
#define ERROR 1e-7
#endif

int test_ldlnp_inv(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, C, C1, D;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    DTYPE n0, n1;
    int ok;
    const char *fact = (flags & ARMAS_LOWER) ? "LDL^T" : "UDU^T";
    const char *blk = lb == 0 ? "unblk" : "  blk";

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&C1, N, N);
    armas_x_diag(&D, &C1, 0);
    armas_x_madd(&D, 1.0, 0, &conf);

    armas_x_set_values(&A0, unitrand, ARMAS_SYMM);
    armas_x_mcopy(&A1, &A0, 0, &conf);
    armas_x_make_trm(&A0, flags);

    if (N < 10) {
        printf("A:\n");
        armas_x_printf(stdout, "%6.3f", &A0);
    }

    env->lb = lb;
    armas_x_ldlfactor(&A0, ARMAS_NOPIVOT, flags, &conf);
    armas_x_ldlinverse(&A0, ARMAS_NOPIVOT, flags, &conf);

    armas_x_mult_sym(ZERO, &C, ONE, &A0, &A1, flags | ARMAS_LEFT, &conf);

    n0 = rel_error(&n1, &C, &C1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * ERROR);
    if (N < 10) {
        printf("A*A.-1 - I:\n");
        armas_x_printf(stdout, "%6.3f", &C);
    }
    printf("%s : %s.(%s).-1*A = I\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));


    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&C);
    armas_x_release(&C1);
    return 1 - ok;
}

int test_ldl_inv(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, C, C1, D;
    armas_pivot_t P0;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    DTYPE n0, n1;
    int ok;
    const char *fact = (flags & ARMAS_LOWER) ? "LDL^T" : "UDU^T";
    const char *blk = (lb == 0) ? "unblk" : "  blk";

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&C1, N, N);
    armas_x_diag(&D, &C1, 0);
    armas_x_madd(&D, 1.0, 0, &conf);
    armas_pivot_init(&P0, N);

    armas_x_set_values(&A0, unitrand, ARMAS_SYMM);
    armas_x_mcopy(&A1, &A0, 0, &conf);
    armas_x_make_trm(&A0, flags);

    if (N < 10) {
        printf("A:\n");
        armas_x_printf(stdout, "%6.3f", &A0);
    }

    env->lb = lb;
    armas_x_ldlfactor(&A0, &P0, flags, &conf);
    armas_x_ldlinverse(&A0, &P0, flags, &conf);

    armas_x_mult_sym(0.0, &C, 1.0, &A0, &A1, flags | ARMAS_LEFT, &conf);

    n0 = rel_error(&n1, &C, &C1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * ERROR);
    if (N < 10) {
        printf("P0: ");
        armas_pivot_printf(stdout, "%d", &P0);
        printf("A*A.-1 - I:\n");
        armas_x_printf(stdout, "%6.3f", &C);
    }
    printf("%s : %s.(%s).-1*A = I\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));


    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&C);
    armas_x_release(&C1);
    armas_pivot_release(&P0);
    return 1 - ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 151;
    int LB = 48;
    int verbose = 0;

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
    } else if (optind < argc - 1) {
        N = atoi(argv[optind]);
        LB = N > 32 ? 24 : 8;
    }

    int fails = 0;

    fails += test_ldlnp_inv(N, 0, ARMAS_LOWER, verbose);
    fails += test_ldlnp_inv(N, LB, ARMAS_LOWER, verbose);
    fails += test_ldlnp_inv(N, 0, ARMAS_UPPER, verbose);
    fails += test_ldlnp_inv(N, LB, ARMAS_UPPER, verbose);
    fails += test_ldl_inv(N, 0, ARMAS_LOWER, verbose);
    fails += test_ldl_inv(N, LB, ARMAS_LOWER, verbose);
    fails += test_ldl_inv(N, 0, ARMAS_UPPER, verbose);
    fails += test_ldl_inv(N, LB, ARMAS_UPPER, verbose);

    exit(fails);
}
