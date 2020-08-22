
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "rqfact"

/*  ---------------------------------------------------------------------------
 *  Test unblk.QR(A) == blk.QR(A)
 *     OK: ||unblk.QR(A) - blk.QR(A)||_1 < N*epsilon
 */
int test_factor(int M, int N, int lb, int verbose)
{
    armas_x_dense_t A0, A1, tau0, tau1;
    DTYPE n0, n1;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();

    if (lb == 0)
        lb = 4;

    armas_x_init(&A0, M, N);
    armas_x_init(&A1, M, N);
    armas_x_init(&tau0, imin(M, N), 1);
    armas_x_init(&tau1, imin(M, N), 1);

    // set source data
    armas_x_set_values(&A0, unitrand, ARMAS_ANY);
    armas_x_mcopy(&A1, &A0, 0, &conf);

    // factorize
    env->lb = 0;
    armas_x_rqfactor(&A0, &tau0, &conf);

    env->lb = lb;
    armas_x_rqfactor(&A1, &tau1, &conf);

    n0 = rel_error((DTYPE *) 0, &A1, &A0, ARMAS_NORM_ONE, 0, &conf);
    n1 = rel_error((DTYPE *) 0, &tau1, &tau0, ARMAS_NORM_TWO, 0, &conf);

    printf("%s: unblk.RQ(A) == blk.RQ(A)\n", PASS(isOK(n0, N) && isOK(n1, N)));
    if (verbose > 0) {
        printf("  || error.RQ  ||_1: %e [%d]\n", n0, ndigits(n0));
        printf("  || error.tau ||_2: %e [%d]\n", n1, ndigits(n1));
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&tau0);
    armas_x_release(&tau1);
    return isOK(n0, N) && isOK(n1, N);
}


int main(int argc, char **argv)
{
    int opt;
    int N = 787;
    int M = 741;
    int LB = 48;
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
