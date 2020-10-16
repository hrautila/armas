
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "trdsec"

#if FLOAT32
// note: with [beta=1e-2, divisor=100.0] fails with initial beta [nan values] (TODO??)
#define DIVISOR 10.0
#define BETA 1e-3
#define SIZE 414
#else
#define DIVISOR 1000.0
#define BETA 1e-3
#define SIZE 414
#endif

// test diagonal:
//    D    = [1.0, 2-(N/2)*beta, 2-(N/2-1)*beta, ... 2+(N/2-1)*beta, 2+(N/2)*beta, 10/3]
//    w    = [2.0, beta, ..., beta, 2.0]
//    Z    = w / ||w||_2
//    rho  = 1.0/||w||_2
int test1(int N, double beta, int verbose)
{
    armas_dense_t D0, Z0, delta0, D1, Z1, delta1, Q, II, sI;
    DTYPE rho, w, nrm;
    int i, ok, fails;
    armas_conf_t conf = *armas_conf_default();

    // make N even
    N += (N & 0x1);
    fails = 0;

    armas_init(&D0, N, 1);
    armas_init(&D1, N, 1);
    armas_init(&Z0, N, 1);
    armas_init(&Z1, N, 1);
    armas_init(&delta0, N, 1);
    armas_init(&delta1, N, 1);
    armas_init(&Q, N, N);
    armas_init(&II, N, N);

    armas_set_at(&D0, 0, 1.0);
    armas_set_at(&Z0, 0, 2.0);
    for (i = 1; i < N - 1; i += 1) {
        if (i < N / 2) {
            armas_set_at(&D0, i, 2.0 - (N / 2 - i + 1) * beta);
        } else {
            armas_set_at(&D0, i, 2.0 + (i + 1 - N / 2) * beta);
        }
        armas_set_at(&Z0, i, beta);
    }
    armas_set_at(&D0, N - 1, 10.0 / 3.0);
    armas_set_at(&Z0, N - 1, 2.0);

    w = armas_nrm2(&Z0, (armas_conf_t *) 0);
    armas_scale(&Z0, ONE/w, (armas_conf_t *) 0);

    rho = 1.0 / (w * w);
    if (verbose > 2 && N <= 10) {
        printf("rho: %e\n", rho);
        printf("D0:\n");
        armas_printf(stdout, "%13e", &D0);
        printf("Z0:\n");
        armas_printf(stdout, "%13e", &Z0);
    }

    armas_trdsec_solve_vec(&D1, &Z1, &Q, &D0, &Z0, rho, &conf);
    armas_trdsec_eigen(&Q, &Z1, &Q, &conf);
    armas_mult(ZERO, &II, ONE, &Q, &Q, ARMAS_TRANSA, &conf);
    if (verbose > 2 && N <= 10) {
        printf("D1:\n");
        armas_printf(stdout, "%13e", &D1);
        armas_axpy(&Z1, -1.0, &Z0, &conf);
        printf("Z0-Z1:\n");
        armas_printf(stdout, "%13e", &Z1);
        printf("I:\n");
        armas_printf(stdout, "%13e", &II);
    }

    armas_diag(&sI, &II, 0);
    armas_madd(&sI, -ONE, 0, &conf);
    nrm = armas_mnorm(&II, ARMAS_NORM_ONE, &conf);
    ok = isOK(nrm, N);
    printf("%s: I == Q.T*Q\n", PASS(ok));
    if (verbose > 0)
        printf("  N=%d, beta=%13e || rel error ||_1: %e [%d]\n", N, beta, nrm,
               ndigits(nrm));
    if (!ok)
        fails++;

    armas_release(&D0);
    armas_release(&D1);
    armas_release(&Z0);
    armas_release(&Z1);
    armas_release(&delta0);
    armas_release(&Q);
    armas_release(&II);

    return fails;
}



int main(int argc, char **argv)
{
    int opt;
    int N = SIZE;
    int verbose = 1;
    DTYPE beta = BETA;

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
    beta /= DIVISOR;
    if (test1(N, beta, verbose))
        fails++;
    beta /= DIVISOR;
    if (test1(N, beta, verbose))
        fails++;

    exit(fails);
}
