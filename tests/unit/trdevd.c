
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

int set_diagonals(armas_x_dense_t *A, int flags, int type, DTYPE coeff)
{
    armas_x_dense_t sD, sE0, sE1;
    int k, N, S, E;

    armas_x_init(&sD, 0, 0);
    armas_x_init(&sE0, 0, 0);
    armas_x_init(&sE1, 0, 0);

    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE0, A, 1);
    armas_x_diag(&sE1, A, -1);
    N = armas_x_size(&sD);

    switch (type) {
    case 2:
        // larger values in the middle
        S = E = 0;
        for (k = 0; k < N-1; k++) {
            if (k & 0x1) {
                armas_x_set_at(&sD, N-1-E, coeff*(k+1));
                E++;
            } else {
                armas_x_set_at(&sD, S, coeff*(k+1));
                S++;
            }
            if (k < N-1) {
                armas_x_set_at(&sE0, k, 1.0);
                armas_x_set_at(&sE1, k, 1.0);
            }
        }
        break;
    case 1:
        // larger values on top
        for (k = 0; k < N-1; k++) {
            armas_x_set_at(&sD, N-1-k, (k+1)*coeff);
            if (k < N-1) {
                armas_x_set_at(&sE0, k, 1.0);
                armas_x_set_at(&sE1, k, 1.0);
            }
        }
        break;
    default:
        // larger values on bottom
        for (k = 0; k < N; k++) {
            armas_x_set_at(&sD, k, (k+1)*coeff);
            if (k < N-1) {
                armas_x_set_at(&sE0, k, 1.0);
                armas_x_set_at(&sE1, k, 1.0);
            }
        }
    }
    return 0;
}

// D0 = |D0| - |D1|
void abs_minus(armas_x_dense_t *D0, armas_x_dense_t *D1)
{
    int k;
    DTYPE tmp;
    for (k = 0; k < armas_x_size(D0); k++) {
        tmp = fabs(armas_x_get_at(D0, k)) - fabs(armas_x_get_at(D1, k));
        armas_x_set_at(D0, k, tmp);
    }
}

// test: 
int test_eigen(int N, int type, DTYPE coeff, int verbose)
{
    armas_x_dense_t A0, V, D, E, sD, sE0, sE1, C, W;
    armas_conf_t conf = *armas_conf_default();
    DTYPE nrm;
    int ok, fails = 0;
    char desc[6] = "typeX";
    desc[4] = '0' + type;
        
    armas_x_init(&A0, N, N);
    set_diagonals(&A0, 0, type, coeff);

    armas_x_init(&D, N, 1);
    armas_x_init(&E, N-1, 1);
    armas_x_diag(&sD, &A0, 0);
    armas_x_diag(&sE0, &A0, 1);
    armas_x_diag(&sE1, &A0, -1);
    armas_x_copy(&D, &sD, &conf);
    armas_x_copy(&E, &sE0, &conf);

    // unit singular vectors
    armas_x_init(&V, N, N);
    armas_x_diag(&sD, &V, 0);
    armas_x_madd(&sD, 1.0, 0);
    
    armas_x_init(&C, N, N);
    armas_x_init(&W, 4*N, 1);

    armas_x_trdeigen(&D, &E, &V, &W, ARMAS_WANTV, &conf);
    // compute: U.T*A*V
    armas_x_mult(&C, &V, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_x_mult(&A0, &C, &V, 1.0, 0.0, ARMAS_NOTRANS, &conf);

    if (verbose > 2 && N < 10) {
        printf("D:\n"); armas_x_printf(stdout, "%6.3f", &D);
        printf("V.T*A*V:\n"); armas_x_printf(stdout, "%6.3f", &A0);
    }

    // compute ||V.T*A*V - S|| (D is column vector, sD is row vector)
    armas_x_diag(&sD, &A0, 0);
    armas_x_axpy(&sD, &D, -1.0, &conf);
    nrm = armas_x_mnorm(&sD, ARMAS_NORM_TWO, &conf);
    ok = isFINE(nrm, N*__ERROR_EVD);
    printf("%s: [%s] V.T*A*V == eigen(A)\n", PASS(ok), desc);
    if (verbose > 0)
        printf("  N=%d || rel error ||_1: %e [%d]\n", N, nrm, ndigits(nrm));

    if (!ok)
        fails++;

    // compute ||I - V*V.T||_1
    armas_x_mult(&C, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_x_diag(&sD, &C, 0);
    armas_x_madd(&sD, -1.0, 0);

    nrm = armas_x_mnorm(&C, ARMAS_NORM_ONE, &conf);
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
    DTYPE coeff = 1.0;

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
