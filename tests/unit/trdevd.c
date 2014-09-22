
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "trdevd"

int set_diagonals(armas_d_dense_t *A, int flags, int type, double coeff)
{
    armas_d_dense_t sD, sE0, sE1;
    int k, N, S, E;

    armas_d_diag(&sD, A, 0);
    armas_d_diag(&sE0, A, 1);
    armas_d_diag(&sE1, A, -1);
    N = armas_d_size(&sD);

    switch (type) {
    case 2:
        // larger values in the middle
        S = E = 0;
        for (k = 0; k < N-1; k++) {
            if (k & 0x1) {
                armas_d_set_at(&sD, N-1-E, coeff*(k+1));
                E++;
            } else {
                armas_d_set_at(&sD, S, coeff*(k+1));
                S++;
            }
            if (k < N-1) {
                armas_d_set_at(&sE0, k, 1.0);
                armas_d_set_at(&sE1, k, 1.0);
            }
        }
        break;
    case 1:
        // larger values on top
        for (k = 0; k < N-1; k++) {
            armas_d_set_at(&sD, N-1-k, (k+1)*coeff);
            if (k < N-1) {
                armas_d_set_at(&sE0, k, 1.0);
                armas_d_set_at(&sE1, k, 1.0);
            }
        }
        break;
    default:
        // larger values on bottom
        for (k = 0; k < N; k++) {
            armas_d_set_at(&sD, k, (k+1)*coeff);
            if (k < N-1) {
                armas_d_set_at(&sE0, k, 1.0);
                armas_d_set_at(&sE1, k, 1.0);
            }
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

// test: 
int test_eigen(int N, int type, double coeff, int verbose)
{
    armas_d_dense_t A0, At, U, V, D, E, sD, sE0, sE1, C, W, t;
    armas_conf_t conf = *armas_conf_default();
    double nrm;
    int ok, fails = 0, err;
    char desc[6] = "typeX";
    desc[4] = '0' + type;
        
    armas_d_init(&A0, N, N);
    set_diagonals(&A0, 0, type, coeff);

    armas_d_init(&D, N, 1);
    armas_d_init(&E, N-1, 1);
    armas_d_diag(&sD, &A0, 0);
    armas_d_diag(&sE0, &A0, 1);
    armas_d_diag(&sE1, &A0, -1);
    armas_d_copy(&D, &sD, &conf);
    armas_d_copy(&E, &sE0, &conf);

    // unit singular vectors
    armas_d_init(&V, N, N);
    armas_d_diag(&sD, &V, 0);
    armas_d_madd(&sD, 1.0, 0);
    
    armas_d_init(&C, N, N);
    armas_d_init(&W, 4*N, 1);

    armas_d_trdeigen(&D, &E, &V, &W, ARMAS_WANTV, &conf);
    // compute: U.T*A*V
    armas_d_mult(&C, &V, &A0, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_mult(&A0, &C, &V, 1.0, 0.0, ARMAS_NOTRANS, &conf);

    if (verbose > 2 && N < 10) {
        printf("D:\n"); armas_d_printf(stdout, "%6.3f", &D);
        printf("V.T*A*V:\n"); armas_d_printf(stdout, "%6.3f", &A0);
    }

    // compute ||V.T*A*V - S|| (D is column vector, sD is row vector)
    armas_d_diag(&sD, &A0, 0);
    armas_d_axpy(&sD, &D, -1.0, &conf);
    nrm = armas_d_mnorm(&sD, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*1e-10);
    printf("%s: [%s] V.T*A*V == eigen(A)\n", PASS(ok), desc);
    if (verbose > 0)
        printf("  N=%d ||V.T*A*V - eigen(A)||_1: %e\n", N, nrm);

    if (!ok)
        fails++;

    // compute ||I - V*V.T||_1
    armas_d_mult(&C, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_diag(&sD, &C, 0);
    armas_d_madd(&sD, -1.0, 0);

    nrm = armas_d_mnorm(&C, ARMAS_NORM_ONE, &conf);
    ok = isFINE(nrm, N*1e-15);
    printf("%s: I == V*V.T\n", PASS(ok));
    if (verbose > 0)
        printf("  N=%d ||I - V*V.T||_1: %e\n", N, nrm);
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
    double coeff = 1.0;

    while ((opt = getopt(argc, argv, "P:c:v")) != -1) {
        switch (opt) {
        case 'P':
            nproc = atoi(optarg);
            break;
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
    if (test_eigen(N, 0, coeff, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
