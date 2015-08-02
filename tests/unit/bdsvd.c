
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "bdsvd"

int set_diagonals(armas_d_dense_t *A, int flags, int type)
{
    armas_d_dense_t sD, sE;
    int k, N, S, E;

    armas_d_diag(&sD, A, 0);
    armas_d_diag(&sE, A, (flags & ARMAS_LOWER ? -1 : 1));
    N = armas_d_size(&sD);

    switch (type) {
    case 2:
        // bidiagonal larger values in the middle
        S = E = 0;
        for (k = 0; k < N-1; k++) {
            if (k & 0x1) {
                armas_d_set_at(&sD, N-1-E, k+1);
                E++;
            } else {
                armas_d_set_at(&sD, S, k+1);
                S++;
            }
            if (k < N-1)
                armas_d_set_at(&sE, k, 1.0);
        }
        break;
    case 1:
        // bidiagonal larger values on top
        for (k = 0; k < N-1; k++) {
            armas_d_set_at(&sD, N-1-k, k+1);
            if (k < N-1)
                armas_d_set_at(&sE, k, 1.0);
        }
        break;
    default:
        // bidiagonal larger values on bottom
        for (k = 0; k < N; k++) {
            armas_d_set_at(&sD, k, k+1);
            if (k < N-1)
                armas_d_set_at(&sE, k, 1.0);
        }
    }
}

// D0 = |D0| - |D1|
void abs_minus(armas_d_dense_t *D0, armas_d_dense_t *D1)
{
    int k;
    double tmp;
    for (k = 0; k < armas_d_size(D0); k++) {
        tmp = fabs(armas_d_get_at(D0, k)) - fabs(armas_d_get_at(D1, k));
        armas_d_set_at(D0, k, tmp);
    }
}

// test: M >= N
int test_tall(int M, int N, int flags, int type, int verbose)
{
    armas_d_dense_t A0, At, U, V, D, E, sD, sE, C, W;
    armas_conf_t conf = *armas_conf_default();
    double nrm, nrm_A;
    int ok, fails = 0;
    char *desc = flags & ARMAS_LOWER ? "lower" : "upper";

    armas_d_init(&A0, M, N);
    set_diagonals(&A0, flags, type);
    nrm_A = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    armas_d_submatrix(&At, &A0, 0, 0, N, N);

    armas_d_init(&D, N, 1);
    armas_d_init(&E, N-1, 1);
    armas_d_diag(&sD, &A0, 0);
    armas_d_diag(&sE, &A0, (flags & ARMAS_LOWER ? -1 : 1));
    armas_d_copy(&D, &sD, &conf);
    armas_d_copy(&E, &sE, &conf);

    // unit singular vectors
    armas_d_init(&U, M, N);
    armas_d_diag(&sD, &U, 0);
    armas_d_madd(&sD, 1.0, 0);

    armas_d_init(&V, N, N);
    armas_d_diag(&sD, &V, 0);
    armas_d_madd(&sD, 1.0, 0);
    
    armas_d_init(&C, N, N);
    armas_d_init(&W, 4*N, 1);

    armas_d_bdsvd(&D, &E, &U, &V, &W, flags|ARMAS_WANTU|ARMAS_WANTV, &conf);

    // compute: U.T*A*V
    armas_d_mult(&C, &U, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_mult(&At, &C, &V, 1.0, 0.0, ARMAS_TRANSB, &conf);

    if (verbose > 2 && N < 10) {
        printf("D:\n"); armas_d_printf(stdout, "%6.3f", &D);
        printf("At:\n"); armas_d_printf(stdout, "%6.3f", &At);
    }

    // compute ||U.T*A*V - S|| (D is column vector, sD is row vector)
    armas_d_diag(&sD, &At, 0);
    abs_minus(&sD, &D);
    nrm = armas_d_mnorm(&sD, ARMAS_NORM_ONE, &conf) / nrm_A;
    ok = isFINE(nrm, N*1e-10);
    printf("%s: [%s] U.T*A*V == S\n", PASS(ok), desc);
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));

    if (!ok)
        fails++;

    // compute: ||I - U.T*U||
    armas_d_mult(&C, &U, &U, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_diag(&sD, &C, 0);
    armas_d_madd(&sD, -1.0, 0);

    nrm = armas_d_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*1e-15);
    printf("%s: I == U.T*U\n", PASS(ok));
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));
    if (!ok)
        fails++;


    // compute ||I - V*V.T||_1
    armas_d_mult(&C, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_madd(&sD, -1.0, 0);

    nrm = armas_d_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*1e-15);
    printf("%s: I == V*V.T\n", PASS(ok));
    if (verbose > 0)
        printf("  M=%d, N=%d || rel error ||_1: %e [%d]\n", M, N, nrm, ndigits(nrm));
    if (!ok)
        fails++;

    return fails;
}

main(int argc, char **argv)
{
    int opt;
    int M = 213;
    int N = 199;
    int K = N;
    int LB = 32;
    int ok = 0;
    int nproc = 1;
    int verbose = 0;

    while ((opt = getopt(argc, argv, "P:v")) != -1) {
        switch (opt) {
        case 'P':
            nproc = atoi(optarg);
            break;
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [M N LB]\n", NAME);
            exit(1);
        }
    }
    
    if (optind < argc-2) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
        LB = atoi(argv[optind+2]);
    } else if (optind < argc-1) {
        N = atoi(argv[optind]);
        M = N;
        LB = atoi(argv[optind+1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        M = N; LB = 0;
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
