
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#if 0
#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t armas_x_dense_t;
typedef float DTYPE;

#define armas_x_init       armas_s_init
#define armas_x_set_values armas_s_set_values
#define armas_x_mult       armas_s_mult
#define armas_x_mult_trm   armas_s_mult_trm
#define armas_x_solve_trm  armas_s_solve_trm
#define armas_x_transpose  armas_s_transpose
#define armas_x_release    armas_s_release
#define armas_x_mcopy      armas_s_mcopy
#define armas_x_printf     armas_s_printf

#define __SCALING (DTYPE)((1 << 14) + 1)
#define STRTOF(arg)  strtof(arg, (char **)0);

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t armas_x_dense_t;
typedef double DTYPE;

#define armas_x_init       armas_d_init
#define armas_x_set_values armas_d_set_values
#define armas_x_mult       armas_d_mult
#define armas_x_mult_trm   armas_d_mult_trm
#define armas_x_solve_trm  armas_d_solve_trm
#define armas_x_transpose  armas_d_transpose
#define armas_x_release    armas_d_release
#define armas_x_mcopy      armas_d_mcopy
#define armas_x_printf     armas_d_printf

#define __SCALING (DTYPE)((1 << 27) + 1)
#define STRTOF(arg)  strtod(arg, (char **)0);

#endif
#include "helper.h"
#endif

#include "testing.h"
#if FLOAT32

#define __SCALING (DTYPE)((1 << 14) + 1)
#define STRTOF(arg)  strtof(arg, (char **)0);

#else

#define __SCALING (DTYPE)((1 << 27) + 1)
#define STRTOF(arg)  strtod(arg, (char **)0);

#endif

DTYPE scaledrand(int i, int j)
{
    DTYPE val = unitrand(i, j);
    return val * __SCALING;
}

static DTYPE Aconstant = 1.0;

DTYPE constant(int i, int j)
{
    return Aconstant;
}

DTYPE near_one(int i, int j)
{
    //if ((i & 0x1) == 0 && (j & 0x1) == 0)
    //return 1.0;
    //return (i & 0x1) == 0 || (j & 0x1) == 0 ? 1.0 - 10.*_EPS : 1.0 + 10.0*_EPS;
    return ((i + j) & 0x1) == 0 ? 1.0 - _EPS : 1.0 + _EPS;
}

int test_left_right(int N, int verbose)
{
    int ok;
    armas_x_dense_t A, At, B, Bt;
    DTYPE n0, nrmB;
    armas_conf_t conf = *armas_conf_default();

    armas_x_init(&A, N, N);
    armas_x_init(&At, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&Bt, N, N);

    armas_x_set_values(&A, one, ARMAS_SYMM);
    armas_x_transpose(&At, &A);
    armas_x_set_values(&B, one, ARMAS_ANY);
    armas_x_mult_trm(&B, 1.0, &A, ARMAS_UPPER, ARMAS_ANY);
    armas_x_transpose(&Bt, &B);
    if (N < 10) {
        printf("A\n");
        armas_x_printf(stdout, "%6.3f", &A);
        printf("At\n");
        armas_x_printf(stdout, "%6.3f", &At);
    }
    nrmB = armas_x_mnorm(&B, ARMAS_NORM_INF, &conf);
    // ||k*A.-1*B + (B.T*-k*A.-T).T|| ~ eps
    armas_x_solve_trm(&B, 2.0, &A, ARMAS_LEFT | ARMAS_UPPER, &conf);
    armas_x_solve_trm(&Bt, -2.0, &At, ARMAS_RIGHT | ARMAS_UPPER | ARMAS_TRANS,
                      &conf);
    if (N < 10) {
        printf("B\n");
        armas_x_printf(stdout, "%6.3f", &B);
        printf("Bt\n");
        armas_x_printf(stdout, "%6.3f", &Bt);
    }
    armas_x_scale_plus(1.0, &B, 1.0, &Bt, ARMAS_TRANSB, &conf);
    if (N < 10) {
        printf("B + B.T\n");
        armas_x_printf(stdout, "%6.3f", &B);
    }
    n0 = armas_x_mnorm(&B, ARMAS_NORM_INF, &conf) / nrmB;
    ok = isOK(n0, N) || n0 == 0.0;
    printf("%4s : || k*A.-1*B + (-k*B.T*A.-T).T|| : %e\n", PASS(ok), n0);
    return 1 - ok;
}

int main(int argc, char **argv)
{
    armas_conf_t conf;
    armas_x_dense_t B0, A, B;

    int ok, opt;
    int N = 301;
    int verbose = 1;
    int fails = 0;
    DTYPE alpha = 1.0;
    DTYPE n0, n1;

    while ((opt = getopt(argc, argv, "vC:")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'C':
            Aconstant = STRTOF(optarg);
            break;
        default:
            fprintf(stderr, "usage: trsm [-P nproc] [size]\n");
            exit(1);
        }
    }

    if (optind < argc)
        N = atoi(argv[optind]);

    conf = *armas_conf_default();

    armas_x_init(&A, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&B0, N, N);

    // Upper triangular matrix
    armas_x_set_values(&A, one, ARMAS_UPPER);
    if (N < 10) {
        printf("A:\n");
        armas_x_printf(stdout, "%8.1e", &A);
    }


    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_LEFT, &conf);
    if (N < 10) {
        printf("A*B:\n");
        armas_x_printf(stdout, "%8.1e", &B);
    }
    armas_x_solve_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_LEFT, &conf);
    if (N < 10) {
        printf("A.-1*B:\n");
        armas_x_printf(stdout, "%8.1e", &B);
    }
    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, L|U|N), A, L|U|N)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_RIGHT, &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_RIGHT, &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, R|U|N), A, R|U|N)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_LEFT | ARMAS_TRANSA,
                     &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_LEFT | ARMAS_TRANSA,
                      &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, L|U|T), A, L|U|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_RIGHT | ARMAS_TRANSA,
                     &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_RIGHT | ARMAS_TRANSA,
                      &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, R|U|T), A, R|U|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // Lower triangular matrix
    armas_x_set_values(&A, one, ARMAS_LOWER);

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_LEFT, &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_LEFT, &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, L|L|N), A, L|L|N)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_RIGHT, &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_RIGHT, &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, R|L|N), A, R|L|N)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_LEFT | ARMAS_TRANSA,
                     &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_LEFT | ARMAS_TRANSA,
                      &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, L|L|T), A, L|L|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_RIGHT | ARMAS_TRANSA,
                     &conf);
    armas_x_solve_trm(&B, alpha, &A, ARMAS_LOWER | ARMAS_RIGHT | ARMAS_TRANSA,
                      &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: B = solve_trm(mult_trm(B, A, R|L|T), A, R|L|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    test_left_right(N, verbose);
    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
