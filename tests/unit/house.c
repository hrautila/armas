
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"


// -------------------------------------------------------------------------------------

int test_householder(const char *name, int M, 
                     armas_x_dense_t *x, int verbose, int flags, armas_conf_t *conf)
{
    armas_x_dense_t y, y1, y2, tau, beta, H, H0, d1, w, w2, z, r;
    DTYPE _beta, _tau, n2;
    int ok, fails = 0, sign;

    // create unit diagonal matrix
    armas_x_init(&H0, M, M);
    armas_x_init(&H, M, M);
    armas_x_diag(&d1, &H, 0);
    armas_x_set_values(&d1, one, ARMAS_ANY);

    armas_x_init(&y, M, 1);
    armas_x_submatrix(&y1, &y, 0, 0, 1, 1);
    armas_x_submatrix(&y2, &y, 1, 0, M-1, 1);

    armas_x_init(&w, M, 1);
    armas_x_init(&z, M, 1);
    armas_x_make(&beta, 1, 1, 1, &_beta);
    armas_x_make(&tau , 1, 1, 1, &_tau);

    armas_x_mcopy(&y, x);
    
    armas_x_set(&y1, 0, 0, 1.0);
    armas_x_set(&beta, 0, 0, armas_x_get(x, 0, 0));
    armas_x_house(&beta, &y2, &tau, flags, conf);

    // compute I - tau*v*v^T
    armas_x_mvupdate(&H, -armas_x_get(&tau, 0, 0), &y, &y, conf);
    // compute H*H - I
    armas_x_mult(0.0, &H0, 1.0, &H, &H, 0, conf);
    armas_x_diag(&d1, &H0, 0);
    armas_x_madd(&d1, -1.0, ARMAS_ANY);
    if (verbose && M < 10) {
        printf("%s: H*H (1)\n", name); armas_x_printf(stdout, "%9.2e", &H0);
    }

    // get ||H0 - I||_2, which is relative error
    n2 = armas_x_mnorm(&H0, ARMAS_NORM_INF, conf);
    ok = isOK(n2, M) || n2 == 0.0;
    printf("%s: H*H == I          : %s [%e]\n", name, PASS(ok), n2);
    fails += (1 - ok);

    // compute H*(H*x) == x
    armas_x_mvmult(0.0, &w, 1.0, &H, x, 0, conf);
    // || H*x - w ||/||w||  == sqrt(||Hw||^2 - beta^2)/beta^2
    armas_x_submatrix(&w2, &w, 1, 0, M-1, 1);
    n2 = armas_x_nrm2(&w2, conf);
    if (flags & ARMAS_HHNEGATIVE) {
        sign = '-';
        n2 = hypot(n2, _beta + armas_x_get(&w, 0, 0))/__ABS(_beta);
    } else {
        sign = ' ';
        n2 = hypot(n2, _beta - armas_x_get(&w, 0, 0))/__ABS(_beta);
    }
    ok = isOK(n2, M) || n2 == 0.0;
    // check signs of beta vs alpha
    if (flags & ARMAS_NONNEG) {
        if (_beta < 0.0) 
            ok = 0;
    } else {
        if (flags & ARMAS_HHNEGATIVE) { 
            if ( __SIGN(_beta) != __SIGN(armas_x_get(x, 0, 0)))
                ok = 0;
        }
        else  {
            if ( __SIGN(_beta) == __SIGN(armas_x_get(x, 0, 0)))
                ok = 0;
        }
    }
    printf("%s: Hx == [%cbeta,0]^T : %s [%e]\n", name, sign, PASS(ok), n2);
    fails += (1 - ok);

    armas_x_mvmult(0.0, &z, 1.0, &H, &w, 0, conf);
    armas_x_axpy(&z, -1.0, x, conf);
    n2 = armas_x_nrm2(&z, conf)/armas_x_nrm2(x, conf);;
    ok = isOK(n2, M) || n2 == 0.0;
    printf("%s: H*(H*x) == x      : %s [%e]\n", name, PASS(ok), n2);
    fails += (1 - ok);
    
    armas_x_release(&H0);
    armas_x_release(&H);
    armas_x_release(&y);
    armas_x_release(&w);
    armas_x_release(&z);
    return fails;
}

int test_unitrand(int M, int verbose)
{
    armas_x_dense_t x;
    DTYPE alpha;
    int flags, fails = 0;
    armas_conf_t conf = *armas_conf_default();

    armas_x_init(&x, M, 1);
    armas_x_set_values(&x, zeromean, ARMAS_ANY);
    alpha = __ABS(armas_x_get(&x, 0, 0));

    // compute for Hx = [beta; 0] : x[0] > 0
    armas_x_set(&x, 0, 0, alpha);
    fails += test_householder("house > ", M, &x, verbose, 0, &conf);
    // compute for Hx = [beta; 0] : x[0] < 0
    armas_x_set(&x, 0, 0, -alpha);
    fails += test_householder("house < ", M, &x, verbose, 0, &conf);
    // compute for Hx = [beta; 0] : beta >= 0 && x[0] > 0
    flags = ARMAS_NONNEG;
    armas_x_set(&x, 0, 0, alpha);
    fails += test_householder("house >+", M, &x, verbose, flags, &conf);
    // compute for Hx = [beta; 0] : beta >= 0 && x[0] < 0
    armas_x_set(&x, 0, 0, -alpha);
    fails += test_householder("house <+", M, &x, verbose, flags, &conf);

    // compute for Hx = [-beta; 0] : x[0] > 0
    flags = ARMAS_HHNEGATIVE;
    armas_x_set(&x, 0, 0, alpha);
    fails += test_householder("house-> ", M, &x, verbose, flags, &conf);
    // compute for Hx = [-beta; 0] : x[0] < 0
    armas_x_set(&x, 0, 0, alpha);
    fails += test_householder("house-< ", M, &x, verbose, flags, &conf);
    // compute for Hx = [-beta; 0] : beta >= 0 && x[0] > 0
    flags = ARMAS_HHNEGATIVE|ARMAS_NONNEG;
    armas_x_set(&x, 0, 0, alpha);
    fails += test_householder("house->+", M, &x, verbose, flags, &conf);
    // compute for Hx = [-beta; 0] : beta >= 0 && x[0] < 0
    armas_x_set(&x, 0, 0, alpha);
    fails += test_householder("house-<+", M, &x, verbose, flags, &conf);

    return fails;
}

DTYPE tiny(int i, int j)
{
    DTYPE safmin = __SAFEMIN/__EPS;
    return unitrand(i, j)*safmin/(i+1);
}

int test_underflow(int M, int verbose)
{
    armas_x_dense_t x, x2;
    armas_conf_t conf = *armas_conf_default();
    int fails = 0;
    
    armas_x_init(&x, M, 1);
    armas_x_submatrix(&x2, &x, 1, 0, M-1, 1);
    
    armas_x_set_values(&x, tiny, ARMAS_ANY);
    fails += test_householder("underflow.L", M, &x, verbose, 0, &conf);
    fails += test_householder("underflow.N", M, &x, verbose, ARMAS_NONNEG, &conf);
    return fails;
}

DTYPE minus_one(int i, int j)
{
    return -1.0;
}

int test_hyperbolic(const char *name, int M, 
                    armas_x_dense_t *x, int verbose, int flags, armas_conf_t *conf)
{
    armas_x_dense_t y, y1, y2, tau, beta, H, H0, H1, d1, w, w2, z, r, J;
    DTYPE _beta, _tau, n2;
    int ok, fails = 0, sign;

    // create unit diagonal matrix
    armas_x_init(&H0, M, M);
    armas_x_init(&H1, M, M);
    armas_x_init(&H, M, M);
    armas_x_diag(&d1, &H, 0);
    armas_x_set_values(&d1, one, ARMAS_ANY);

    // J = diag(1, -1, -1, ... -1)
    armas_x_init(&J, M, 1);
    armas_x_set_values(&J, minus_one, ARMAS_ANY);
    armas_x_set(&J, 0, 0, 1.0);
    
    armas_x_init(&y, M, 1);
    armas_x_submatrix(&y1, &y, 0, 0, 1, 1);
    armas_x_submatrix(&y2, &y, 1, 0, M-1, 1);

    armas_x_init(&w, M, 1);
    armas_x_init(&z, M, 1);
    armas_x_make(&beta, 1, 1, 1, &_beta);
    armas_x_make(&tau , 1, 1, 1, &_tau);

    armas_x_mcopy(&y, x);
    
    armas_x_set(&y1, 0, 0, 1.0);
    armas_x_set(&beta, 0, 0, armas_x_get(x, 0, 0));

    if (armas_x_hhouse(&beta, &y2, &tau, flags, conf) < 0)
        printf("hyper error\n");

    // compute I - tau*J*v*v^T
    armas_x_mvupdate(&H0, -armas_x_get(&tau, 0, 0), &y, &y, conf);
    armas_x_mult_diag(&H0, &J, 1.0, ARMAS_LEFT, conf);
    armas_x_add_elems(&H0, &H, ARMAS_ANY);
    armas_x_mcopy(&H1, &H0);

    // compute H = H*J*H^T - J ; H1 = H, H0 = H*J
    armas_x_mult_diag(&H0, &J, 1.0, ARMAS_RIGHT, conf);
    armas_x_mult(0.0, &H, 1.0, &H0, &H1, ARMAS_TRANSB, conf);
    armas_x_diag(&d1, &H, 0);
    armas_x_axpy(&d1, -1.0, &J, conf);

    if (verbose > 2 && M < 10) {
        printf("%s: H \n", name); armas_x_printf(stdout, "%9.2e", &H1);
        printf("%s: H*J\n", name); armas_x_printf(stdout, "%9.2e", &H0);
        printf("%s: H*J*H^T - J\n", name); armas_x_printf(stdout, "%9.2e", &H);
    }

    // get ||H - J||_2, which is relative error
    n2 = armas_x_mnorm(&H, ARMAS_NORM_INF, conf);
    ok = isOK(n2, M) || n2 == 0.0;
    printf("%s: H*J*H^T == J      : %s [%e]\n", name, PASS(ok), n2);
    fails += (1 - ok);

    // compute (H*x) == [beta; 0]^T
    armas_x_mvmult(0.0, &w, 1.0, &H1, x, 0, conf);
    if (verbose && M < 10) {
        printf("   H*x"); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&r, &w));
    }
    // || H*x - w ||/||w||  == sqrt(||Hw||^2 - beta^2)/beta^2
    armas_x_submatrix(&w2, &w, 1, 0, M-1, 1);
    n2 = armas_x_nrm2(&w2, conf);
    if (flags & ARMAS_HHNEGATIVE) {
        // Hx = [-beta; 0]
        sign = '-';
        n2 = hypot(n2, -_beta - armas_x_get(&w, 0, 0))/__ABS(_beta);
    } else {
        // Hx = [beta; 0]
        sign = ' ';
        n2 = hypot(n2, _beta - armas_x_get(&w, 0, 0))/__ABS(_beta);
    }
    ok = isOK(n2, M) || n2 == 0.0;
    if ((flags & ARMAS_NONNEG) && _beta < 0.0) {
        if (verbose) {
            printf("  beta: %9.2e\n", _beta);
        }
        ok = 0;
    } 
    printf("%s: Hx == [%cbeta,0]^T : %s [%e]\n", name, sign, PASS(ok), n2);
    fails += (1 - ok);

    // compute H^T*J*H*x
    armas_x_mcopy(&H0, &H1);
    armas_x_mult_diag(&H0, &J, 1.0, ARMAS_LEFT, conf);
    armas_x_mvmult(0.0, &w, 1.0, &H0,  x, 0, conf);
    armas_x_mvmult(0.0, &z, 1.0, &H1, &w, ARMAS_TRANS, conf);
    if (verbose && M < 10) {
        printf("        x "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&r, x));
        printf("H^T*J*H*x "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&r, &z));
    }
    armas_x_set(&z, 0, 0, - armas_x_get(&z, 0, 0));
    armas_x_axpy(&z, 1.0, x, conf);
    n2 = armas_x_nrm2(&z, conf)/armas_x_nrm2(x, conf);;
    ok = isOK(n2, M) || n2 == 0.0;
    printf("%s: H^T*J*H*x == J*x  : %s [%e]\n", name, PASS(ok), n2);
    fails += (1 - ok);

    armas_x_release(&H0);
    armas_x_release(&H1);
    armas_x_release(&H);
    armas_x_release(&y);
    armas_x_release(&w);
    armas_x_release(&z);
    armas_x_release(&J);
    return fails;
}

int test_unitrand_hyper(int M, int verbose)
{
    armas_x_dense_t x;
    DTYPE nrmx;
    int flags, fails = 0;
    armas_conf_t conf = *armas_conf_default();
    armas_x_init(&x, M, 1);
    armas_x_set_values(&x, zeromean, ARMAS_ANY);
    nrmx = armas_x_nrm2(&x, &conf);

    armas_x_set(&x, 0, 0, 2*nrmx);
    // for [beta; 0] : x[0]>0
    fails += test_hyperbolic("hyper > ", M, &x, verbose, 0, &conf);
    // for [beta; 0] : x[0]<0
    armas_x_set(&x, 0, 0, -2*nrmx);
    fails += test_hyperbolic("hyper < ", M, &x, verbose, 0, &conf);
    // for [beta; 0] : beta >= 0, x[0] > 0
    flags = ARMAS_NONNEG;
    armas_x_set(&x, 0, 0, 2*nrmx);
    fails += test_hyperbolic("hyper >+", M, &x, verbose, flags, &conf);
    // for [beta; 0] : beta >= 0,  x[0] < 0
    armas_x_set(&x, 0, 0, -2*nrmx);
    fails += test_hyperbolic("hyper <+", M, &x, verbose, flags, &conf);
    // for [-beta; 0] : x[0]>0
    flags = ARMAS_HHNEGATIVE;
    armas_x_set(&x, 0, 0, 2*nrmx);
    fails += test_hyperbolic("hyper-> ", M, &x, verbose, flags, &conf);
    // for [-beta; 0] : x[0]<0
    armas_x_set(&x, 0, 0, -2*nrmx);
    fails += test_hyperbolic("hyper-< ", M, &x, verbose, flags, &conf);
    // for [-beta; 0] : beta >= 0, x[0]>0
    flags = ARMAS_HHNEGATIVE|ARMAS_NONNEG;
    armas_x_set(&x, 0, 0, 2*nrmx);
    fails += test_hyperbolic("hyper->+", M, &x, verbose, flags, &conf);
    // for [-beta; 0] : beta >= 0, x[0]<0
    armas_x_set(&x, 0, 0, -2*nrmx);
    fails += test_hyperbolic("hyper-<+", M, &x, verbose, flags, &conf);
    return fails;
}

double _two(int i, int j)
{
    return 2.0;
}

int test_hyper(int M, int verbose)
{
    armas_x_dense_t x, x2;
    DTYPE nrmx;
    int fails = 0;
    armas_conf_t conf = *armas_conf_default();

    // beta^2 = sqrt(alpha^2 - normx^2)  alpha=5, normx=4->x_i=2 ==> beta = 3.0
    M = 5;
    armas_x_init(&x, M, 1);
    armas_x_subvector(&x2, &x, 1, M-1);
    armas_x_set_values(&x, _two, ARMAS_ANY);
    nrmx = armas_x_nrm2(&x2, &conf);
    //armas_x_set(&x, 0, 0, __SQRT(4.0+nrmx*nrmx));
    armas_x_set(&x, 0, 0, 5.0);
    fails += test_hyperbolic("hyper ", M, &x, verbose, 0, &conf);
    return fails;
}


int main(int argc, char **argv)
{
    int opt;
    int M = 10;
    int N = 9;
    int K = N;
    int ok = 0;
    int verbose = 0;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: prog [-v] [M N LB]\n");
            exit(1);
        }
    }
    
    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        M = atoi(argv[optind]);
        N = M;
    }

    int fails = 0;
    fails += test_unitrand(M, verbose);
    fails += test_underflow(M, verbose);
    fails += test_unitrand_hyper(M, verbose);
    fails += test_hyper(M, verbose);
    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
