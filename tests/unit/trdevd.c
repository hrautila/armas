
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR     1e-6
#define __ERROR_EVD 1e-4
#else
#define __ERROR     1e-14
#define __ERROR_EVD 1e-10
#endif

#define NAME "trdevd"

int set_diagonals(__Matrix *A, int flags, int type, __Dtype coeff)
{
    __Matrix sD, sE0, sE1;
    int k, N, S, E;

    matrix_init(&sD, 0, 0);
    matrix_init(&sE0, 0, 0);
    matrix_init(&sE1, 0, 0);

    matrix_diag(&sD, A, 0);
    matrix_diag(&sE0, A, 1);
    matrix_diag(&sE1, A, -1);
    N = matrix_size(&sD);

    switch (type) {
    case 2:
        // larger values in the middle
        S = E = 0;
        for (k = 0; k < N-1; k++) {
            if (k & 0x1) {
                matrix_set_at(&sD, N-1-E, coeff*(k+1));
                E++;
            } else {
                matrix_set_at(&sD, S, coeff*(k+1));
                S++;
            }
            if (k < N-1) {
                matrix_set_at(&sE0, k, 1.0);
                matrix_set_at(&sE1, k, 1.0);
            }
        }
        break;
    case 1:
        // larger values on top
        for (k = 0; k < N-1; k++) {
            matrix_set_at(&sD, N-1-k, (k+1)*coeff);
            if (k < N-1) {
                matrix_set_at(&sE0, k, 1.0);
                matrix_set_at(&sE1, k, 1.0);
            }
        }
        break;
    default:
        // larger values on bottom
        for (k = 0; k < N; k++) {
            matrix_set_at(&sD, k, (k+1)*coeff);
            if (k < N-1) {
                matrix_set_at(&sE0, k, 1.0);
                matrix_set_at(&sE1, k, 1.0);
            }
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

// test: 
int test_eigen(int N, int type, __Dtype coeff, int verbose)
{
    __Matrix A0, V, D, E, sD, sE0, sE1, C, W;
    armas_conf_t conf = *armas_conf_default();
    __Dtype nrm;
    int ok, fails = 0;
    char desc[6] = "typeX";
    desc[4] = '0' + type;
        
    matrix_init(&A0, N, N);
    set_diagonals(&A0, 0, type, coeff);

    matrix_init(&D, N, 1);
    matrix_init(&E, N-1, 1);
    matrix_diag(&sD, &A0, 0);
    matrix_diag(&sE0, &A0, 1);
    matrix_diag(&sE1, &A0, -1);
    matrix_copy(&D, &sD, &conf);
    matrix_copy(&E, &sE0, &conf);

    // unit singular vectors
    matrix_init(&V, N, N);
    matrix_diag(&sD, &V, 0);
    matrix_madd(&sD, 1.0, 0);
    
    matrix_init(&C, N, N);
    matrix_init(&W, 4*N, 1);

    matrix_trdeigen(&D, &E, &V, &W, ARMAS_WANTV, &conf);
    // compute: U.T*A*V
    matrix_mult(&C, &V, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    matrix_mult(&A0, &C, &V, 1.0, 0.0, ARMAS_NOTRANS, &conf);

    if (verbose > 2 && N < 10) {
        printf("D:\n"); matrix_printf(stdout, "%6.3f", &D);
        printf("V.T*A*V:\n"); matrix_printf(stdout, "%6.3f", &A0);
    }

    // compute ||V.T*A*V - S|| (D is column vector, sD is row vector)
    matrix_diag(&sD, &A0, 0);
    matrix_axpy(&sD, &D, -1.0, &conf);
    nrm = matrix_mnorm(&sD, ARMAS_NORM_TWO, &conf);
    ok = isFINE(nrm, N*__ERROR_EVD);
    printf("%s: [%s] V.T*A*V == eigen(A)\n", PASS(ok), desc);
    if (verbose > 0)
        printf("  N=%d || rel error ||_1: %e [%d]\n", N, nrm, ndigits(nrm));

    if (!ok)
        fails++;

    // compute ||I - V*V.T||_1
    matrix_mult(&C, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    matrix_diag(&sD, &C, 0);
    matrix_madd(&sD, -1.0, 0);

    nrm = matrix_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*__ERROR);
    printf("%s: I == V*V.T\n", PASS(ok));
    if (verbose > 0)
        printf("  N=%d || rel error ||_1: %e [%d]\n", N, nrm, ndigits(nrm));
    if (!ok)
        fails++;

    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 199;
    int verbose = 0;
    __Dtype coeff = 1.0;

    while ((opt = getopt(argc, argv, "c:v")) != -1) {
        switch (opt) {
        case 'c':
            coeff = atof(optarg);
            break;
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [M N LB]\n", NAME);
            exit(1);
        }
    }
    
    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;
    if (test_eigen(N, 0, coeff, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
