
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqfact"

/*  ---------------------------------------------------------------------------
 *  Test unblk.QR(A) == blk.QR(A)
 *     OK: ||unblk.QR(A) - blk.QR(A)||_1 < N*epsilon
 */
int test_factor(int M, int N, int lb, int verbose)
{
    armas_dense_t A0, A1, tau0, tau1;
    DTYPE n0, n1;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();

    if (lb == 0)
        lb = 4;

    armas_init(&A0, M, N);
    armas_init(&A1, M, N);
    armas_init(&tau0, imin(M, N), 1);
    armas_init(&tau1, imin(M, N), 1);

    // set source data
    armas_set_values(&A0, unitrand, 0);
    armas_mcopy(&A1, &A0, 0, &conf);

    // factorize
    env->lb = 0;
    armas_lqfactor(&A0, &tau0, &conf);

    env->lb = lb;
    armas_lqfactor(&A1, &tau1, &conf);

    n0 = rel_error((DTYPE *) 0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    n1 = rel_error((DTYPE *) 0, &tau0, &tau1, ARMAS_NORM_TWO, ARMAS_NONE,
                   &conf);

    printf("%s: unblk.LQ(A) == blk.LQ(A)\n", PASS(isOK(n0, N) && isOK(n1, N)));
    if (verbose > 0) {
        printf("  || error.LQ  ||_1: %e [%d]\n", n0, ndigits(n0));
        printf("  || error.tau ||_2: %e [%d]\n", n1, ndigits(n1));
    }

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&tau0);
    armas_release(&tau1);
    return isOK(n0, N) && isOK(n1, N);
}


int main(int argc, char **argv)
{
    int opt;
    int N = 787;
    int M = 741;
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
    if (!test_factor(M, N, LB, verbose))
        fails++;

    exit(fails);
}
