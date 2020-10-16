
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "lqsolve"

// test least squares min || B - A.T*X ||, M < N
//   1. compute B = A.T*X0
//   2. compute B =  A.-1*B
//   3. compute || X0 - B || == O(eps)
int test_lss(int M, int N, int K, int lb, int verbose)
{
    armas_dense_t A0, tau0;
    armas_dense_t B0, X0, X;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();
    int ok;
    DTYPE nrm;

    armas_init(&A0, M, N);
    armas_init(&B0, N, K);
    armas_init(&X0, M, K);
    armas_init(&tau0, M, 1);

    // set source data
    armas_set_values(&A0, unitrand, ARMAS_ANY);

    // set initial X
    armas_set_values(&X0, unitrand, ARMAS_ANY);

    // compute: B0 = A0.T*X0
    armas_mult(ZERO, &B0, ONE, &A0, &X0, ARMAS_TRANSA, &conf);

    env->lb = lb;
    // factor
    armas_lqfactor(&A0, &tau0, &conf);

    // solve B0 = A.-T*B0
    armas_lqsolve(&B0, &A0, &tau0, ARMAS_TRANS, &conf);

    // X0 = X0 - A.-1*B0
    armas_submatrix(&X, &B0, 0, 0, M, K);

    nrm = rel_error((DTYPE *) 0, &X, &X0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isOK(nrm, N);
    printf("%s: min || B - A.T*X ||\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
    armas_release(&A0);
    armas_release(&X0);
    armas_release(&B0);
    armas_release(&tau0);
    return ok;
}

// test: min || X || s.t. A*X = B
int test_min(int M, int N, int K, int lb, int verbose)
{
    armas_dense_t A0, A1, tau0;
    armas_dense_t B0, X0, B;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();
    int ok;
    DTYPE nrm, nrm0;

    armas_init(&A0, M, N);
    armas_init(&A1, M, N);
    armas_init(&B0, N, K);
    armas_init(&X0, N, K);
    armas_init(&tau0, M, 1);

    // set source data
    armas_set_values(&A0, unitrand, ARMAS_ANY);
    armas_mcopy(&A1, &A0, 0, &conf);

    // set B0
    armas_set_values(&B0, unitrand, ARMAS_ANY);
    nrm0 = armas_mnorm(&B0, ARMAS_NORM_ONE, &conf);

    env->lb = lb;

    // factor
    armas_lqfactor(&A0, &tau0, &conf);

    // X0 = A.-T*B0
    armas_mcopy(&X0, &B0, 0, &conf);
    if (armas_lqsolve(&X0, &A0, &tau0, ARMAS_NONE, &conf) < 0)
        printf("solve error: %d\n", conf.error);

    // B = B - A*X
    armas_submatrix(&B, &B0, 0, 0, M, K);
    armas_mult(ONE, &B, -ONE, &A1, &X0, ARMAS_NONE, &conf);

    nrm = armas_mnorm(&B, ARMAS_NORM_ONE, &conf) / nrm0;
    ok = isOK(nrm, N);
    printf("%s: min || X || s.t. A*X = B\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error || : %e [%d]\n", nrm, ndigits(nrm));
    }
    armas_release(&A0);
    armas_release(&A1);
    armas_release(&X0);
    armas_release(&B0);
    armas_release(&tau0);

    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 787;
    int N = 741;
    int K = N;
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
    // assert(M >= N)
    if (M < N) {
        int t = M;
        M = N;
        N = t;
    }

    int fails = 0;
    K = N / 2;
    if (!test_lss(N, M, K, LB, verbose))
        fails++;
    if (!test_min(N, M, K, LB, verbose))
        fails++;
    exit(fails);
}
