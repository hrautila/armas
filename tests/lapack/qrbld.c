
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "qrbld"


/*  -----------------------------------------------------------------------------------
 *  Test 5: unblk.Q(qr(A)) == blk.Q(qr(A))
 *    OK: ||unblk.Q(qr(A)) - blk.Q(qr(A))||_1 ~~ n*eps
 */
int test_qrbuild(int M, int N, int K, int lb, int verbose)
{
    char ct = N == K ? 'N' : 'K';
    armas_dense_t A0, A1, tau0;
    int ok;
    DTYPE n0, n1;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();

    armas_init(&A0, M, N);
    armas_init(&A1, M, N);
    armas_init(&tau0, imin(M, N), 1);

    // set source data
    armas_set_values(&A0, unitrand, ARMAS_ANY);

    // factorize
    env->lb = lb;
    armas_qrfactor(&A0, &tau0, &conf);
    armas_mcopy(&A1, &A0, 0, &conf);

    // compute Q = buildQ(qr(A))
    env->lb = 0;
    armas_qrbuild(&A0, &tau0, K, &conf);
    env->lb = lb;
    armas_qrbuild(&A1, &tau0, K, &conf);

    n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isOK(n0, N);
    printf("%s: unblk.Q(qr(A),%c) == blk.Q(qr(A),%c)\n", PASS(ok), ct, ct);
    if (verbose > 0) {
        printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
    }

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&tau0);

    return ok;
}

/*  -----------------------------------------------------------------------------------
 *  Test 6: Q(qr(A)).T * Q(qr(A)) == I 
 *    OK: ||I - Q.T*Q||_1 ~~ n*eps
 */
int test_qrbuild_identity(int M, int N, int K, int lb, int verbose)
{
    const char *blk = (lb > 0) ? "  blk" : "unblk";
    char ct = N == K ? 'N' : 'K';
    armas_dense_t A0, C0, C1, tau0, D;
    int ok;
    DTYPE n0, n1;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();

    armas_init(&A0, M, N);
    armas_init(&C0, N, N);
    armas_init(&C1, N, N);
    armas_init(&tau0, imin(M, N), 1);
    armas_set_values(&C1, zero, ARMAS_ANY);
    armas_diag(&D, &C1, 0);
    armas_madd(&D, ONE, 0, &conf);

    // set source data
    armas_set_values(&A0, unitrand, ARMAS_ANY);

    // factorize
    env->lb = lb;
    armas_qrfactor(&A0, &tau0, &conf);

    // compute Q = buildQ(qr(A)), K first columns
    env->lb = lb;
    armas_qrbuild(&A0, &tau0, K, &conf);

    // C0 = Q.T*Q 
    armas_mult(ZERO, &C0, ONE, &A0, &A0, ARMAS_TRANSA, &conf);

    n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isOK(n0, N);
    printf("%s: %s Q(qr(A),%c).T * Q(qr(A),%c) == I\n", PASS(ok), blk, ct, ct);
    if (verbose > 0) {
        printf("  || rel error ||_1: %e [%d]\n", n0, ndigits(n0));
    }
    armas_release(&A0);
    armas_release(&C0);
    armas_release(&C1);
    armas_release(&tau0);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 787;
    int N = 741;
    int LB = 64;
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

    if (optind < argc - 2) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
        LB = atoi(argv[optind + 2]);
    } else if (optind < argc - 1) {
        N = atoi(argv[optind]);
        M = N;
        LB = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        M = N;
        LB = 0;
    }

    int fails = 0;
    if (!test_qrbuild(M, N, N, LB, verbose))
        fails += 1;
    if (!test_qrbuild(M, N, N / 2, LB, verbose))
        fails += 1;
    if (!test_qrbuild_identity(M, N, N, 0, verbose))
        fails += 1;
    if (!test_qrbuild_identity(M, N, N / 2, 0, verbose))
        fails += 1;
    if (!test_qrbuild_identity(M, N, N, LB, verbose))
        fails += 1;
    if (!test_qrbuild_identity(M, N, N / 2, LB, verbose))
        fails += 1;

    exit(fails);
}
