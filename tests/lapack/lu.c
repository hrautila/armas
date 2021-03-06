
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "test"
#if FLOAT32
#define MAX_ERROR 8e-6
#else
#define MAX_ERROR 1e-12
#endif

int test_solve(int M, int N, int lb, int verbose)
{
    armas_dense_t A0, A1;
    armas_dense_t B0, X0;
    armas_pivot_t P0;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();
    const char *blk = (lb == 0) ? "unblk" : "blk";
    int ok;
    DTYPE nrm, nrm0;

    armas_init(&A0, N, N);
    armas_init(&A1, N, N);
    armas_init(&B0, N, M);
    armas_init(&X0, N, M);
    armas_pivot_init(&P0, N);

    // set source data
    armas_set_values(&A0, unitrand, ARMAS_ANY);
    armas_mcopy(&A1, &A0, 0, &conf);

    armas_set_values(&B0, unitrand, ARMAS_ANY);
    armas_mcopy(&X0, &B0, 0, &conf);
    nrm0 = armas_mnorm(&B0, ARMAS_NORM_ONE, &conf);

    env->lb = lb;
    armas_lufactor(&A0, &P0, &conf);

    // solve
    armas_lusolve(&X0, &A0, &P0, ARMAS_NONE, &conf);

    // B0 = B0 - A*X0
    armas_mult(1.0, &B0, -1.0, &A1, &X0, ARMAS_NONE, &conf);
    nrm = armas_mnorm(&B0, ARMAS_NORM_ONE, &conf) / nrm0;

    ok = isFINE(nrm, N * MAX_ERROR);

    printf("%s: A*(%s.LU(A).1*B) == B\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
    return ok;
}

int check_lufactor(DTYPE *nrm, armas_dense_t *LU, armas_dense_t *T, armas_pivot_t *P, armas_dense_t *ref)
{
    armas_dense_t D;
    armas_conf_t *cf = armas_conf_default();
    armas_set_values(T, zero, ARMAS_ANY);
    armas_diag(&D, T, 0);
    armas_set_values(&D, one, ARMAS_ANY);
    // I * L * U
    armas_mult_trm(T, 1.0, LU, ARMAS_RIGHT|ARMAS_LOWER|ARMAS_UNIT, cf);
    armas_mult_trm(T, 1.0, LU, ARMAS_RIGHT|ARMAS_UPPER, cf);
    if (ref && nrm) {
        if (P)
            armas_pivot_rows(T, P, ARMAS_PIVOT_BACKWARD, cf);
        *nrm =  rel_error((DTYPE *)0, T, ref, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    }
    return 1;
}

int test_factor(int M, int N, int lb, int verbose)
{
    armas_dense_t A, A0, A1;
    armas_pivot_t P0, P1;
    armas_env_t *env = armas_getenv();
    armas_conf_t conf = *armas_conf_default();
    int ok;
    DTYPE nrm;

    armas_init(&A,  M, N);
    armas_init(&A0, M, N);
    armas_init(&A1, M, N);
    armas_pivot_init(&P0, N);
    armas_pivot_init(&P1, N);

    // set source data
    armas_set_values(&A, unitrand, ARMAS_ANY);
    armas_mcopy(&A0, &A, 0, &conf);

    //armas_lufactor(&A0, &P0, &conf);
    env->lb = 0;
    armas_lufactor(&A0, &P0, &conf);
    check_lufactor(&nrm, &A0, &A1, &P0, &A);

    ok = isOK(nrm, N);
    printf("%s: unblk: I*LU == A\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }

    env->lb = lb;
    armas_mcopy(&A1, &A, 0, &conf);
    armas_lufactor(&A1, &P1, &conf);

    nrm = rel_error((DTYPE *) 0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isOK(nrm, N);

    printf("%s: unblk.LU(A) == blk.LU(A)\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }

    check_lufactor(&nrm, &A1, &A0, &P0, &A);
    ok = isOK(nrm, N);
    printf("%s:  blk: I*LU == A\n", PASS(ok));
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }

    armas_release(&A0);
    armas_release(&A1);
    armas_pivot_release(&P0);
    armas_pivot_release(&P1);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 411;
    int N = 411;
    int LB = 32;
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

    // unblocked
    if (!test_solve(M, N, 0, verbose))
        fails++;

    // blocked
    if (!test_solve(M, N, LB, verbose))
        fails++;

    exit(fails);
}
