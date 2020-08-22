
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

int set_diagonals(armas_x_dense_t * A, int flags, int type)
{
    armas_x_dense_t sD, sE;
    unsigned int k, N, S, E;
    DTYPE v;

    armas_x_init(&sD, 0, 0);
    armas_x_init(&sE, 0, 0);
    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, A, (flags & ARMAS_LOWER ? -1 : 1));
    N = armas_x_size(&sD);

    switch (type) {
    case 2:
        // bidiagonal larger values in the middle
        S = E = 0;
        for (k = 0; k < N - 1; k++) {
            if (k & 0x1) {
                armas_x_set_at(&sD, N - 1 - E, k + 1);
                E++;
            } else {
                armas_x_set_at(&sD, S, k + 1);
                S++;
            }
            if (k < N - 1)
                armas_x_set_at(&sE, k, 1.0);
        }
        break;
    case 1:
        // bidiagonal larger values on top
        for (k = 0; k < N - 1; k++) {
            armas_x_set_at(&sD, N - 1 - k, k + 1);
            if (k < N - 1)
                armas_x_set_at(&sE, k, 1.0);
        }
        break;
    default:
        // bidiagonal larger values on bottom
        v = 1.0;
        for (k = 0; k < N; k++) {
            armas_x_set_at(&sD, k, (v + 1.0));
            if (k < N - 1)
                armas_x_set_at(&sE, k, 1.0);
        }
    }
    return 0;
}

// D0 = |D0| - |D1|
void abs_minus(armas_x_dense_t * D0, armas_x_dense_t * D1)
{
    unsigned int k, N;
    DTYPE tmp;
    N = (int) armas_x_size(D0);
    for (k = 0; k < N; k++) {
        tmp = fabs(armas_x_get_at(D0, k)) - fabs(armas_x_get_at(D1, k));
        armas_x_set_at(D0, k, tmp);
    }
}

// test: M >= N
int test_tall(int M, int N, int flags, int type, int verbose)
{
    armas_x_dense_t A0, At, U, V, D, E, sD, sE, C;
    armas_conf_t conf = *armas_conf_default();
    DTYPE nrm, nrm_A;
    armas_wbuf_t wb;
    int ok, fails = 0;
    char *desc = flags & ARMAS_LOWER ? "lower" : "upper";

    armas_x_init(&A0, M, N);
    set_diagonals(&A0, flags, type);
    nrm_A = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    armas_x_submatrix(&At, &A0, 0, 0, N, N);

    armas_x_init(&D, N, 1);
    armas_x_init(&E, N - 1, 1);
    armas_x_diag(&sD, &A0, 0);
    armas_x_diag(&sE, &A0, (flags & ARMAS_LOWER) ? -1 : 1);
    armas_x_copy(&D, &sD, &conf);
    armas_x_copy(&E, &sE, &conf);

    // unit singular vectors
    armas_x_init(&U, M, N);
    armas_x_diag(&sD, &U, 0);
    armas_x_madd(&sD, ONE, 0, &conf);

    armas_x_init(&V, N, N);
    armas_x_diag(&sD, &V, 0);
    armas_x_madd(&sD, ONE, 0, &conf);

    armas_x_init(&C, N, N);
    armas_walloc(&wb, 4 * N * sizeof(DTYPE));

    //armas_x_bdsvd_w(&D, &E, &U, &V, flags|ARMAS_WANTU|ARMAS_WANTV, &wb, &conf);
    armas_x_bdsvd(&D, &E, &U, &V, flags | ARMAS_WANTU | ARMAS_WANTV, &conf);

    // compute: U.T*A*V
    armas_x_mult(ZERO, &C, ONE, &U, &A0, ARMAS_TRANSA, &conf);
    armas_x_mult(ZERO, &At, ONE, &C, &V, ARMAS_TRANSB, &conf);

    if (verbose > 2 && N < 10) {
        printf("D:\n");
        armas_x_printf(stdout, "%6.3f", &D);
        printf("At:\n");
        armas_x_printf(stdout, "%6.3f", &At);
    }
    // compute ||U.T*A*V - S|| (D is column vector, sD is row vector)
    armas_x_diag(&sD, &At, 0);
    abs_minus(&sD, &D);
    nrm = armas_x_mnorm(&sD, ARMAS_NORM_ONE, &conf) / nrm_A;
    ok = isFINE(nrm, N * __ERROR);
    printf("%s: [%s] U.T*A*V == S\n", PASS(ok), desc);
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm,
               ndigits(nrm));

    if (!ok)
        fails++;

    // compute: ||I - U.T*U||
    armas_x_mult(ZERO, &C, ONE, &U, &U, ARMAS_TRANSA, &conf);
    armas_x_diag(&sD, &C, 0);
    armas_x_madd(&sD, -ONE, 0, &conf);

    nrm = armas_x_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N * __ERROR);
    printf("%s: I == U.T*U\n", PASS(ok));
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm,
               ndigits(nrm));
    if (!ok)
        fails++;


    // compute ||I - V*V.T||_1
    armas_x_mult(ZERO, &C, ONE, &V, &V, ARMAS_TRANSA, &conf);
    armas_x_madd(&sD, -ONE, 0, &conf);

    nrm = armas_x_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N * __ERROR);
    printf("%s: I == V*V.T\n", PASS(ok));
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm,
               ndigits(nrm));
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

    if (optind < argc - 1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
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
