
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR 1e-5
#else
#define __ERROR 1e-13
#endif

#define NAME "bdsvd"

int set_diagonals(__Matrix *A, int flags, int type)
{
    __Matrix sD, sE;
    int k, N, S, E;

    matrix_init(&sD, 0, 0);
    matrix_init(&sE, 0, 0);
    matrix_diag(&sD, A, 0);
    matrix_diag(&sE, A, (flags & ARMAS_LOWER ? -1 : 1));
    N = matrix_size(&sD);

    switch (type) {
    case 2:
        // bidiagonal larger values in the middle
        S = E = 0;
        for (k = 0; k < N-1; k++) {
            if (k & 0x1) {
                matrix_set_at(&sD, N-1-E, k+1);
                E++;
            } else {
                matrix_set_at(&sD, S, k+1);
                S++;
            }
            if (k < N-1)
                matrix_set_at(&sE, k, 1.0);
        }
        break;
    case 1:
        // bidiagonal larger values on top
        for (k = 0; k < N-1; k++) {
            matrix_set_at(&sD, N-1-k, k+1);
            if (k < N-1)
                matrix_set_at(&sE, k, 1.0);
        }
        break;
    default:
        // bidiagonal larger values on bottom
        for (k = 0; k < N; k++) {
            matrix_set_at(&sD, k, k+1);
            if (k < N-1)
                matrix_set_at(&sE, k, 1.0);
        }
    }
    return 0;
}

// D0 = |D0| - |D1|
void abs_minus(__Matrix *D0, __Matrix *D1)
{
    int k;
    __Dtype tmp;
    for (k = 0; k < matrix_size(D0); k++) {
        tmp = fabs(matrix_get_at(D0, k)) - fabs(matrix_get_at(D1, k));
        matrix_set_at(D0, k, tmp);
    }
}

// test: M >= N
int test_tall(int M, int N, int flags, int type, int verbose)
{
    __Matrix A0, At, U, V, D, E, sD, sE, C, W;
    armas_conf_t conf = *armas_conf_default();
    __Dtype nrm, nrm_A;
    int ok, fails = 0;
    char *desc = flags & ARMAS_LOWER ? "lower" : "upper";

    matrix_init(&A0, M, N);
    set_diagonals(&A0, flags, type);
    nrm_A = matrix_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    matrix_submatrix(&At, &A0, 0, 0, N, N);

    matrix_init(&D, N, 1);
    matrix_init(&E, N-1, 1);
    matrix_diag(&sD, &A0, 0);
    matrix_diag(&sE, &A0, (flags & ARMAS_LOWER ? -1 : 1));
    matrix_copy(&D, &sD, &conf);
    matrix_copy(&E, &sE, &conf);

    // unit singular vectors
    matrix_init(&U, M, N);
    matrix_diag(&sD, &U, 0);
    matrix_madd(&sD, 1.0, 0);

    matrix_init(&V, N, N);
    matrix_diag(&sD, &V, 0);
    matrix_madd(&sD, 1.0, 0);
    
    matrix_init(&C, N, N);
    matrix_init(&W, 4*N, 1);

    matrix_bdsvd(&D, &E, &U, &V, &W, flags|ARMAS_WANTU|ARMAS_WANTV, &conf);

    // compute: U.T*A*V
    matrix_mult(&C, &U, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    matrix_mult(&At, &C, &V, 1.0, 0.0, ARMAS_TRANSB, &conf);

    if (verbose > 2 && N < 10) {
        printf("D:\n"); matrix_printf(stdout, "%6.3f", &D);
        printf("At:\n"); matrix_printf(stdout, "%6.3f", &At);
    }

    // compute ||U.T*A*V - S|| (D is column vector, sD is row vector)
    matrix_diag(&sD, &At, 0);
    abs_minus(&sD, &D);
    nrm = matrix_mnorm(&sD, ARMAS_NORM_ONE, &conf) / nrm_A;
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: [%s] U.T*A*V == S\n", PASS(ok), desc);
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));

    if (!ok)
        fails++;

    // compute: ||I - U.T*U||
    matrix_mult(&C, &U, &U, 1.0, 0.0, ARMAS_TRANSA, &conf);
    matrix_diag(&sD, &C, 0);
    matrix_madd(&sD, -1.0, 0);

    nrm = matrix_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: I == U.T*U\n", PASS(ok));
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));
    if (!ok)
        fails++;


    // compute ||I - V*V.T||_1
    matrix_mult(&C, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    matrix_madd(&sD, -1.0, 0);

    nrm = matrix_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: I == V*V.T\n", PASS(ok));
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));
    if (!ok)
        fails++;

    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 213;
    int N = 199;
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
    
    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        M = N;
    }

    int fails = 0;
    if (test_tall(M, N, ARMAS_UPPER, 0, verbose))
        fails++;
    if (test_tall(M, N, ARMAS_LOWER, 0, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
