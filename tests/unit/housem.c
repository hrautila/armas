
// Copyright (c) Harri Rautila, 2017

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
    armas_d_dense_t x, y, z, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();

    // x and y 
    armas_d_init(&x, m, 1);
    armas_d_init(&y, m, p);
    armas_d_init(&z, m, p);
    armas_d_init(&tau, n, 1);
    armas_d_init(&w, p, 1);
    armas_d_init(&P, m, n);
    armas_d_set_values(&P, unitrand, 0);
    armas_d_set_values(&y, unitrand, 0);
    armas_d_mcopy(&z, &y);
    
    // generate Householder reflectors
    for (int k = 0; k < n; k++) {
        armas_x_submatrix(&x0, &P, k, k, 1, 1);
        armas_x_submatrix(&x1, &P, k+1, k, m-k-1, 1);
        //armas_x_set(&x0, 0, 0, -armas_x_get(&x0, 0, 0));
        armas_x_submatrix(&t, &tau, k, 0, 1, 1);
        armas_x_house(&x0, &x1, &t,  flags, &cf);
    }

    // Q^T*(Q*y) == y
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_LEFT|ARMAS_TRANS, &cf) < 0)
        printf("left.1: error %d\n", cf.error);
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_LEFT, &cf) < 0)
        printf("left.2: error %d\n", cf.error);

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_ONE, 0, &cf);
    int ok = isFINE(relerr, m*__ERROR);
    char *s = (flags & ARMAS_UNIT) != 0 ? "unit scaled" : "   unscaled";

    printf("%s: (%s) y - Q.T*(Qy) == y\n", PASS(ok), s);
    if (verbose > 0)
        printf("   ||y - Q^T*(Qy)||: %e [%d]\n", relerr, ndigits(relerr));

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&tau);
    armas_d_release(&P);
    armas_d_release(&w);

    return ok;
}

// compute: ||x - (x*Q)*Q^T||
int test_right(int flags, int verbose, int m, int n, int p)
{
    armas_d_dense_t x, y, z, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    // x and y 
    armas_d_init(&x, m, 1);
    armas_d_init(&y, p, m);
    armas_d_init(&z, p, m);
    armas_d_init(&tau, n, 1);
    armas_d_init(&w, p, 1);
    armas_d_init(&P, m, n);
    armas_d_set_values(&P, unitrand, 0);
    armas_d_set_values(&y, unitrand, 0);
    armas_d_mcopy(&z, &y);
    
    // generate householder reflectors
    for (int k = 0; k < n; k++) {
        armas_x_submatrix(&x0, &P, k, k, 1, 1);
        armas_x_submatrix(&x1, &P, k+1, k, m-k-1, 1);
        //armas_x_set(&x0, 0, 0, -armas_x_get(&x0, 0, 0));
        armas_x_submatrix(&t, &tau, k, 0, 1, 1);
        armas_x_house(&x0, &x1, &t,  flags, &cf);
    }

    // (y*Q^T)*Q == y
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_RIGHT|ARMAS_TRANS, &cf) < 0)
        printf("right.1: error %d\n", cf.error);
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_RIGHT, &cf) < 0)
        printf("right.2: error %d\n", cf.error);

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_INF, 0, &cf);
    int ok = isFINE(relerr, m*__ERROR);
    char *s = (flags & ARMAS_UNIT) != 0 ? "unit scaled" : "   unscaled";

    printf("%s: (%s) y - (yQ.T)*Q == y\n", PASS(ok), s);
    if (verbose > 0)
        printf("   ||y - (yQ^T)*Q||: %e [%d]\n", relerr, ndigits(relerr));

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&tau);
    armas_d_release(&P);
    armas_d_release(&w);

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

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
