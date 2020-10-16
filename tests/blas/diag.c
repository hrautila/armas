
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "diag"

static
int test_left(int M, int N, int verbose, armas_conf_t *cf)
{
    armas_dense_t A0, A1, D;
    int ok;
    DTYPE nrm, n1;

    armas_init(&A0, M, N);
    armas_init(&A1, M, N);
    armas_init(&D, M, 1);

    armas_set_values(&A0, unitrand, 0);
    armas_mcopy(&A1, &A0, 0, cf);
    armas_set_values(&D, unitrand, 0);

    armas_mult_diag(&A1, 1.0, &D, ARMAS_LEFT, cf);
    armas_solve_diag(&A1, 1.0, &D, ARMAS_LEFT, cf);

    nrm = rel_error(&n1, &A1, &A0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = isOK(nrm, N);
    printf("%6s: A == D.-1*D*A\n", PASS(ok));
    if (verbose > 0)
        printf("    || rel error ||_1: %e [%d]\n", nrm, ndigits(nrm));

    return 1 - ok;
}

static
int test_right(int M, int N, int verbose, armas_conf_t *cf)
{
    armas_dense_t A0, A1, D;
    int ok;
    DTYPE nrm, n1;

    armas_init(&A0, M, N);
    armas_init(&A1, M, N);
    armas_init(&D, N, 1);

    armas_set_values(&A0, unitrand, 0);
    armas_mcopy(&A1, &A0, 0, cf);
    armas_set_values(&D, unitrand, 0);

    armas_mult_diag(&A1, 1.0, &D, ARMAS_RIGHT, cf);
    armas_solve_diag(&A1, 1.0, &D, ARMAS_RIGHT, cf);

    nrm = rel_error(&n1, &A1, &A0, ARMAS_NORM_ONE, 0, cf);
    ok = isOK(nrm, N);
    printf("%6s: A == A*D*D.-1\n", PASS(ok));
    if (verbose > 0)
        printf("    || rel error ||_1: %e, [%d]\n", nrm, ndigits(nrm));

    return 1 - ok;
}

int main(int argc, char **argv)
{
    armas_conf_t *cf;
    int opt;
    int M = 787;
    int N = 741;
    int verbose = 1;
    int all = 1;
    int left = 0;
    int right = 0;

    cf = armas_conf_default();
    while ((opt = getopt(argc, argv, "vLR")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'L':
            left = 1;
            all = 0;
            break;
        case 'R':
            right = 1;
            all = 0;
            break;
        default:
            fprintf(stderr, "usage: %s [-vLR] [M N]\n", NAME);
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
    if (all) {
        fails += test_left(M, N, verbose, cf);
        fails += test_right(M, N, verbose, cf);
    } else {
        if (left)
            fails += test_left(M, N, verbose, cf);
        if (right)
            fails += test_right(M, N, verbose, cf);
    }

    exit(fails);
}
