
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
/*
#if FLOAT32
#define _ERROR 1e-6
#else
#define _ERROR 1e-14
#endif
*/
#define NAME "evd"

int test_1(int N, int flags, int verbose)
{
    armas_x_dense_t A, A0, D, I0, sD, V, T;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1, nrm_A;
    int err, ok;
    char *uplo = flags & ARMAS_LOWER ? "L" : "U";
    armas_wbuf_t wb = ARMAS_WBNULL;

    armas_x_init(&A, N, N);
    armas_x_init(&A0, N, N);
    armas_x_init(&V, N, N);
    armas_x_init(&I0, N, N);
    armas_x_init(&D, N, 1);

    armas_x_set_values(&A, unitrand, ARMAS_SYMM);
    armas_x_mcopy(&A0, &A, 0, &conf);
    nrm_A = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf);

    if (armas_x_eigen_sym_w(&D, &A, flags | ARMAS_WANTV, &wb, &conf) < 0) {
        printf("eigen_sym: workspace calculation failure\n");
        return 0;
    }
    armas_walloc(&wb, wb.bytes);

    err = armas_x_eigen_sym(&D, &A, flags | ARMAS_WANTV, &conf);
    if (err) {
        printf("err = %d, %d\n", err, conf.error);
        return 0;
    }

    armas_x_diag(&sD, &I0, 0);
    armas_x_mult(ZERO, &I0, ONE, &A, &A, ARMAS_TRANSB, &conf);
    armas_x_madd(&sD, -ONE, 0, &conf);
    n0 = armas_x_mnorm(&I0, ARMAS_NORM_ONE, &conf);

    armas_x_mcopy(&V, &A, 0, &conf);
    armas_x_mult_diag(&V, ONE, &D, ARMAS_RIGHT, &conf);
    armas_x_mult(ONE, &A0, -ONE, &V, &A, ARMAS_TRANSB, &conf);

    if (N < 10 && verbose > 2) {
        printf("D:\n");
        armas_x_printf(stdout, "%6.3f", armas_x_col_as_row(&T, &D));
        printf("I - V.T*V:\n");
        armas_x_printf(stdout, "%6.3f", &I0);
        printf("A - V*D*V.T:\n");
        armas_x_printf(stdout, "%6.3f", &A0);
    }
    n1 = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf);
    n1 /= nrm_A;
    ok = isOK(n1, N);

    printf("%s [N=%d, uplo='%s']: A == V*D*V.T\n", PASS(ok), N, uplo);
    if (verbose > 0) {
        printf("  ||A - V*D*V.T||_1: %e [%d]\n", n1, ndigits(n1));
        printf("  ||I - V.T*V||_1  : %e [%d]\n", n0, ndigits(n0));
    }
    armas_x_release(&A);
    armas_x_release(&A0);
    armas_x_release(&V);
    armas_x_release(&I0);
    armas_x_release(&D);
    armas_wrelease(&wb);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 313;
    int verbose = 1;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [N]\n", NAME);
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;

    if (!test_1(N, ARMAS_LOWER, verbose))
        fails++;

    if (!test_1(N, ARMAS_UPPER, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
