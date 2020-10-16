
// Copyright by libARMAS authors. See AUTHORS file in this archive.

#include <stdio.h>
#include <unistd.h>
#include "testing.h"
#include "internal.h"
#include "partition.h"

#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-14
#endif


// compute: ||x - Q^T*(Q*x)||
int test_left(int flags, int verbose, int m, int n, int p)
{
    armas_dense_t x, y, z, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();

    // x and y 
    armas_init(&x, m, 1);
    armas_init(&y, m, p);
    armas_init(&z, m, p);
    armas_init(&tau, n, 1);
    armas_init(&w, p, 1);
    armas_init(&P, m, n);
    armas_set_values(&P, unitrand, 0);
    armas_set_values(&y, unitrand, 0);
    armas_mcopy(&z, &y, 0, &cf);

    // generate Householder reflectors
    for (int k = 0; k < n; k++) {
        armas_submatrix(&x0, &P, k, k, 1, 1);
        armas_submatrix(&x1, &P, k + 1, k, m - k - 1, 1);
        //armas_set(&x0, 0, 0, -armas_get(&x0, 0, 0));
        armas_submatrix(&t, &tau, k, 0, 1, 1);
        armas_house(&x0, &x1, &t, flags, &cf);
    }

    // Q^T*(Q*y) == y
    int err = armas_housemult(&y, &tau, &P, flags | ARMAS_LEFT | ARMAS_TRANS, &cf);
    if (err < 0)
        printf("left.1: error %d\n", cf.error);
    err = armas_housemult(&y, &tau, &P, flags | ARMAS_LEFT, &cf);
    if (err < 0)
        printf("left.2: error %d\n", cf.error);

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_ONE, 0, &cf);
    int ok = isFINE(relerr, m * __ERROR);
    char *s = (flags & ARMAS_UNIT) != 0 ? "unit scaled" : "   unscaled";

    printf("%s: (%s) y - Q.T*(Qy) == y\n", PASS(ok), s);
    if (verbose > 0)
        printf("   ||y - Q^T*(Qy)||: %e [%d]\n", relerr, ndigits(relerr));

    armas_release(&x);
    armas_release(&y);
    armas_release(&z);
    armas_release(&tau);
    armas_release(&P);
    armas_release(&w);

    return ok;
}

// compute: ||x - (x*Q)*Q^T||
int test_right(int flags, int verbose, int m, int n, int p)
{
    armas_dense_t x, y, z, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    // x and y 
    armas_init(&x, m, 1);
    armas_init(&y, p, m);
    armas_init(&z, p, m);
    armas_init(&tau, n, 1);
    armas_init(&w, p, 1);
    armas_init(&P, m, n);
    armas_set_values(&P, unitrand, 0);
    armas_set_values(&y, unitrand, 0);
    armas_mcopy(&z, &y, 0, &cf);

    // generate householder reflectors
    for (int k = 0; k < n; k++) {
        armas_submatrix(&x0, &P, k, k, 1, 1);
        armas_submatrix(&x1, &P, k + 1, k, m - k - 1, 1);
        //armas_set(&x0, 0, 0, -armas_get(&x0, 0, 0));
        armas_submatrix(&t, &tau, k, 0, 1, 1);
        armas_house(&x0, &x1, &t, flags, &cf);
    }

    // (y*Q^T)*Q == y
    int err = armas_housemult(&y, &tau, &P, flags | ARMAS_RIGHT | ARMAS_TRANS, &cf);
    if (err < 0)
        printf("right.1: error %d\n", cf.error);
    err = armas_housemult(&y, &tau, &P, flags | ARMAS_RIGHT, &cf);
    if  (err < 0)
        printf("right.2: error %d\n", cf.error);

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_INF, 0, &cf);
    int ok = isFINE(relerr, m * __ERROR);
    char *s = (flags & ARMAS_UNIT) != 0 ? "unit scaled" : "   unscaled";

    printf("%s: (%s) y - (yQ.T)*Q == y\n", PASS(ok), s);
    if (verbose > 0)
        printf("   ||y - (yQ^T)*Q||: %e [%d]\n", relerr, ndigits(relerr));

    armas_release(&x);
    armas_release(&y);
    armas_release(&z);
    armas_release(&tau);
    armas_release(&P);
    armas_release(&w);

    return stat;
}

int main(int argc, char **argv)
{
    int opt;
    int verbose = 1;
    int m = 76;
    int n = 8;
    int p = 13;

    while ((opt = getopt(argc, argv, "vn:m:p:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'm':
            m = atoi(optarg);
            break;
        case 'n':
            n = atoi(optarg);
            break;
        case 'p':
            p = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: housem [-v -m M -n N -p P] \n");
            exit(1);
        }
    }

    test_right(0, verbose, m, n, p);
    test_right(ARMAS_UNIT, verbose, m, n, p);
    test_left(0, verbose, m, n, p);
    test_left(ARMAS_UNIT, verbose, m, n, p);
    return 0;
}
