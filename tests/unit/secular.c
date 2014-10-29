
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "trdsec"


// test diagonal:
//    D    = [1.0, 2-(N/2)*beta, 2-(N/2-1)*beta, ... 2+(N/2-1)*beta, 2+(N/2)*beta, 10/3]
//    w    = [2.0, beta, ..., beta, 2.0]
//    Z    = w / ||w||_2
//    rho  = 1.0/||w||_2
int test1(int N, double beta, int verbose) {
    armas_d_dense_t D0, Z0, delta0, D1, Z1, delta1, Q, I, sI;
    double rho, w, nrm;
    int err, i, K, ok, fails;
    armas_conf_t conf = *armas_conf_default();
    
    // make N even
    N += (N & 0x1);
    fails = 0;

    armas_d_init(&D0, N, 1);
    armas_d_init(&D1, N, 1);
    armas_d_init(&Z0, N, 1);
    armas_d_init(&Z1, N, 1);
    armas_d_init(&delta0, N, 1);
    armas_d_init(&delta1, N, 1);
    armas_d_init(&Q, N, N);
    armas_d_init(&I, N, N);
    
    armas_d_set_at(&D0, 0, 1.0);
    armas_d_set_at(&Z0, 0, 2.0);
    for (i = 1; i < N-1; i += 1) {
        if (i < N/2) {
            armas_d_set_at(&D0, i, 2.0 - (N/2-i)*beta);
        } else {
            armas_d_set_at(&D0, i, 2.0 + (i+1-N/2)*beta);
        }
        armas_d_set_at(&Z0, i, beta);
    }
    armas_d_set_at(&D0, N-1, 10.0/3.0);
    armas_d_set_at(&Z0, N-1, 2.0);

    w = armas_d_nrm2(&Z0, (armas_conf_t *)0);
    armas_d_invscale(&Z0, w, (armas_conf_t *)0);
    
    rho = 1.0/(w*w);
    if (verbose > 2 && N <= 10) {
        printf("rho: %e\n", rho);
        printf("D0:\n"); armas_d_printf(stdout, "%13e", &D0);
        printf("Z0:\n"); armas_d_printf(stdout, "%13e", &Z0);
    }

    armas_d_trdsec_solve_vec(&D1, &Z1, &Q, &D0, &Z0, rho, &conf);
    armas_d_trdsec_eigen(&Q, &Z1, &Q, &conf);
    armas_d_mult(&I, &Q, &Q, 1.0, 0.0, ARMAS_TRANSA, &conf);
    if (verbose > 2 && N <= 10) {
        printf("D1:\n"); armas_d_printf(stdout, "%13e", &D1);
        armas_d_axpy(&Z1, &Z0, -1.0, &conf);
        printf("Z0-Z1:\n"); armas_d_printf(stdout, "%13e", &Z1);
        printf("I:\n"); armas_d_printf(stdout, "%13e", &I);
    }

    armas_d_diag(&sI, &I, 0);
    armas_d_madd(&sI, -1.0, 0);
    nrm = armas_d_mnorm(&I, ARMAS_NORM_ONE, &conf);
    ok = isOK(nrm, N);
    printf("%s: I == Q.T*Q\n", PASS(ok));
    if (verbose > 0)
        printf("  N=%d, beta=%13e ||I - Q.T*Q||_1: %e\n", N, beta, nrm);
    if (!ok)
        fails++;

    armas_d_release(&D0);
    armas_d_release(&D1);
    armas_d_release(&Z0);
    armas_d_release(&Z1);
    armas_d_release(&delta0);
    armas_d_release(&Q);
    armas_d_release(&I);

    return fails;
}



main(int argc, char **argv)
{
    int opt;
    int N = 414;
    int ok = 0;
    int verbose = 1;
    double beta = 1e-3;

    while ((opt = getopt(argc, argv, "b:v")) != -1) {
        switch (opt) {
        case 'b':
            beta = atof(optarg);
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
    if (test1(N, beta, verbose))
        fails++;
    beta /= 1000.0;
    if (test1(N, beta, verbose))
        fails++;
    beta /= 1000.0;
    if (test1(N, beta, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
