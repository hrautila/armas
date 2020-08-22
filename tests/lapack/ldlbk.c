
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "ldl"
#if FLOAT32
#define REL_ERROR 8e-6
#else
#define REL_ERROR 1e-11
#endif

int test_factor(int M, int N, int lb, int verbose, int flags)
{
    armas_x_dense_t A0, A1;
    armas_pivot_t P0, P1;
    armas_env_t *env = armas_getenv();
    armas_conf_t cf = *armas_conf_default();
    int ok;
    char uplo = (flags & ARMAS_UPPER) ? 'U' : 'L';
    DTYPE nrm;

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_pivot_init(&P0, N);
    armas_pivot_init(&P1, N);

    // set source data
    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0, 0, &cf);

    cf.error = 0;
    env->lb = 0;
    armas_x_bkfactor(&A0, &P0, flags, &cf);
    //armas_x_bkfactor_w(&A0, &P0, flags, &wb, &cf);
    if (verbose > 1 && cf.error != 0)
        printf("1. error=%d\n", cf.error);

    cf.error = 0;
    env->lb = lb;
    armas_x_bkfactor(&A1, &P1, flags, &cf);
    //armas_x_bkfactor_w(&A1, &P1, flags, &wb, &cf);
    if (verbose > 1 && cf.error != 0)
        printf("1. error=%d\n", cf.error);

    nrm = rel_error((DTYPE *) 0, &A0, &A1, ARMAS_NORM_ONE, 0, &cf);
    ok = isOK(nrm, N);

    printf("%s: unblk.LDL(A,%c) == blk.LDL(A,%c)\n", PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("  || error.LDL(A, '%c') ||: %e [%d]\n", uplo, nrm,
               ndigits(nrm));
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_pivot_release(&P0);
    armas_pivot_release(&P1);

    return ok;
}

int test_solve(int M, int N, int lb, int verbose, int flags)
{
    armas_x_dense_t A0, A1;
    armas_x_dense_t B0, X0;
    armas_pivot_t P0;
    armas_env_t *env = armas_getenv();
    armas_conf_t cf = *armas_conf_default();
    char uplo = (flags & ARMAS_UPPER) ? 'U' : 'L';
    int ok;
    DTYPE nrm, nrm_A;

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&B0, N, M);
    armas_x_init(&X0, N, M);
    armas_pivot_init(&P0, N);

    // set source data
    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0, 0, &cf);

    armas_x_set_values(&B0, unitrand, 0);
    armas_x_mcopy(&X0, &B0, 0, &cf);
    nrm_A = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &cf);

    env->lb = lb;
    armas_x_bkfactor(&A0, &P0, flags, &cf);

    // solve
    armas_x_bksolve(&X0, &A0, &P0, flags, &cf);
    // B0 = B0 - A*X0
    armas_x_mult_sym(1.0, &B0, -1.0, &A1, &X0, ARMAS_LEFT | flags, &cf);
    nrm = armas_x_mnorm(&B0, ARMAS_NORM_ONE, &cf);
    nrm /= nrm_A;

    ok = isFINE(nrm, N * REL_ERROR);

    printf("%s: LDL(%c)  A*(A.-1*B) == B\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf(" || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 657;
    int N = 657;
    int LB = 40;
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
    if (!test_factor(M, N, LB, verbose, ARMAS_LOWER))
        fails++;
    if (!test_factor(M, N, LB, verbose, ARMAS_UPPER))
        fails++;
    if (!test_solve(M, N, LB, verbose, ARMAS_LOWER))
        fails++;
    if (!test_solve(M, N, LB, verbose, ARMAS_UPPER))
        fails++;

    exit(fails);
}
