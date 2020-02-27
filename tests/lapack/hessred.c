
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define ERROR 1e-6
#else
#define ERROR 1e-13
#endif

#define NAME "hessred"

int test_reduce(int M, int N, int lb, int verbose)
{
    armas_x_dense_t A0, A1, tau0, tau1;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();
    int ok;
    DTYPE n0, n1;

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&tau0, N - 1, 1);
    armas_x_init(&tau1, N - 1, 1);

    // set source data
    armas_x_set_values(&A0, unitrand, ARMAS_ANY);
    armas_x_mcopy(&A1, &A0, 0, &conf);

    // unblocked reduction
    env->lb = 0;
    if (armas_x_hessreduce(&A0, &tau0, &conf) < 0)
        printf("unblocked reduce error %d\n", conf.error);

    // blocked reduction
    env->lb = lb;
    if (armas_x_hessreduce(&A1, &tau1, &conf) < 0)
        printf("blocked reduce error: %d\n", conf.error);


    n0 = rel_error((DTYPE *) 0, &A0, &A1, ARMAS_NORM_ONE, 0, &conf);
    n1 = rel_error((DTYPE *) 0, &tau0, &tau1, ARMAS_NORM_TWO, 0, &conf);
    ok = isFINE(n0, N * ERROR);

    printf("%s: unblk.Hess(A) == blk.Hess(A)\n", PASS(ok));
    if (verbose > 0) {
        printf("  || error.Hess ||: %e [%d]\n", n0, ndigits(n0));
        printf("  || error.tau  ||: %e [%d]\n", n1, ndigits(n1));
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&tau0);
    armas_x_release(&tau1);
    return ok;
}

int test_mult(int M, int N, int lb, int verbose)
{
    armas_x_dense_t A0, A1, B, tau0, Blow;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    DTYPE nrm;

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&tau0, N - 1, 1);

    // set source data
    armas_x_set_values(&A0, unitrand, ARMAS_ANY);
    armas_x_mcopy(&A1, &A0, 0, &conf);

    // reduce to Hessenberg matrix
    env->lb = lb;
    if (armas_x_hessreduce(&A0, &tau0, &conf) < 0)
        printf("hess: reduce error %d\n", conf.error);

    // extract B = Hess(A)  
    armas_x_mcopy(&B, &A0, 0, &conf);
    armas_x_submatrix(&Blow, &B, 1, 0, N - 1, N - 1);
    armas_x_make_trm(&Blow, ARMAS_UPPER);

    // A = H*B*H.T; update B with H.T and H
    if (armas_x_hessmult(&B, &A0, &tau0, ARMAS_LEFT, &conf) < 0)
        printf("hessmult left: error %d\n", conf.error);
    if (armas_x_hessmult(&B, &A0, &tau0, ARMAS_RIGHT | ARMAS_TRANS, &conf) < 0)
        printf("hessmult right: error %d\n", conf.error);

    // B == A1?
    nrm = rel_error((DTYPE *) 0, &B, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isOK(nrm, N);
    printf("%s: Q*Hess(A)*Q.T == A\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&B);
    armas_x_release(&tau0);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 787;
    int N = 741;
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

    if (!test_reduce(M, N, LB, verbose))
        fails++;
    if (!test_mult(M, N, LB, verbose))
        fails++;

    exit(fails);
}
