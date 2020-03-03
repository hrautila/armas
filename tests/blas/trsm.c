
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

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

int test_left_right(int N, int verbose, int unit)
{
    int ok;
    armas_x_dense_t A, B, Bt;
    DTYPE n0, nrmB;
    armas_conf_t cf = *armas_conf_default();

    armas_x_init(&A, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&Bt, N, N);

    printf("** trsm: left-and-right, %s\n", unit ? "unit diagonal" : "");

    armas_x_set_values(&A, one, ARMAS_SYMM);
    armas_x_set_values(&B, one, 0);
    armas_x_mult_trm(&B, 1.0, &A, ARMAS_UPPER, 0);
    armas_x_mcopy(&Bt, &B, ARMAS_TRANS, &cf);

    nrmB = armas_x_mnorm(&B, ARMAS_NORM_INF, &cf);
    // ||k*A.-1*B + (B.T*-k*A.-T).T|| ~ eps
    armas_x_solve_trm(&B, 2.0, &A, ARMAS_LEFT | ARMAS_UPPER, &cf);
    armas_x_solve_trm(&Bt, -2.0, &A, ARMAS_RIGHT | ARMAS_UPPER | ARMAS_TRANS, &cf);
    armas_x_mplus(1.0, &B, 1.0, &Bt, ARMAS_TRANS, &cf);

    n0 = armas_x_mnorm(&B, ARMAS_NORM_INF, &cf) / nrmB;
    ok = isOK(n0, N) || n0 == 0.0;
    printf("%6s : k*A^-1*B  ==  (k*B^T*A^-T)^T\n", PASS(ok));
    printf("    || rel error || : %e [%d]\n", n0, ndigits(n0));
    return 1 - ok;
}

int left(int N, int K, int unit, int verbose, armas_conf_t *cf)
{
    armas_d_dense_t X, Y, X0, A;
    DTYPE n0, n1;
    int ok, fails = 0;

    armas_x_init(&Y, N, K);
    armas_x_init(&X0, N, K);
    armas_x_init(&X, N, K);
    armas_x_init(&A, N, N);

    armas_x_set_values(&X0, unitrand, ARMAS_NULL);
    armas_x_set_values(&X, zero, ARMAS_NULL);
    armas_x_set_values(&A, one, ARMAS_SYMM);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }
    armas_x_mcopy(&X, &X0, 0, cf);

    printf("** trsm: left, %s\n", unit ? "unit diagonal" : "");
    // upper
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_UPPER | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_UPPER | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, 10 * N) ? 1 : 0;
    printf("%6s : Left, U|N  : X == trsm(A*X, A)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // upper,trans
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_UPPER | ARMAS_TRANSA | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_UPPER | ARMAS_TRANSA | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Left, U|T  : X == trsm(A^T*X, A^T)\n", PASS(ok));

    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // lower
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_LOWER | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_LOWER | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Left, L|N  : X == trsm(A*X, A)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // lower,trans
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_LOWER | ARMAS_TRANSA | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_LEFT | ARMAS_LOWER | ARMAS_TRANSA | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Left, L|T  : X == trsm(A^T*X, A^T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int right(int N, int K, int unit, int verbose, armas_conf_t *cf)
{
    armas_d_dense_t X, Y, X0, A, At;
    DTYPE n0, n1;
    int ok, fails = 0;

    armas_x_init(&Y, K, N);
    armas_x_init(&X0, K, N);
    armas_x_init(&X, K, N);
    armas_x_init(&A, N, N);
    armas_x_init(&At, N, N);

    armas_x_set_values(&X0, unitrand, ARMAS_NULL);
    armas_x_set_values(&X, zero, ARMAS_NULL);
    armas_x_set_values(&A, one, ARMAS_SYMM);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }

    printf("** trsm: right, %s\n", unit ? "unit diagonal" : "");
    armas_x_mcopy(&X, &X0, 0, cf);
    // upper
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_UPPER | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_UPPER | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Right, U|N : X == trsm(A*X, A)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // upper,trans
    armas_x_mult_trm( &X, 1.0, &A, ARMAS_RIGHT | ARMAS_UPPER | ARMAS_TRANSA | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_UPPER | ARMAS_TRANSA | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Right, U|T : X == trsm(A^T*X, A^T)\n", PASS(ok));

    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // lower
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_LOWER | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_LOWER | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Right, L|N : X == trsm(A*X, A)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // lower,trans
    armas_x_mult_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_LOWER | ARMAS_TRANSA | unit, cf);
    armas_x_solve_trm(&X, 1.0, &A, ARMAS_RIGHT | ARMAS_LOWER | ARMAS_TRANSA | unit, cf);

    n0 = rel_error(&n1, &X, &X0, ARMAS_NORM_INF, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : Right, L|T : X == trsm(A^T*X, A^T)\n", PASS(ok));
    if (verbose > 0)
    {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;

    int opt;
    int N = 301, K = 255;
    int verbose = 1;
    int fails = 0;
    int unit = 0;
    int all = 1;
    int do_left = 0;
    int do_right = 0;

    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vC:r:nb:uLR")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'n':
            cf.optflags |= ARMAS_ONAIVE;
            break;
        case 'b':
            env->mb = atoi(optarg);
            env->nb = env->kb = env->mb;
            cf.optflags = 0;
            break;
        case 'u':
            unit = ARMAS_UNIT;
            break;
        case 'C':
            Aconstant = STRTOF(optarg);
            break;
        case 'L':
            do_left = 1 - do_left;
            all = 0;
            break;
        case 'R':
            do_right = 1 - do_right;
            all = 0;
            break;
        default:
            fprintf(stderr, "usage: trsm [-P nproc] [size]\n");
            exit(1);
        }
    }

    if (optind < argc - 1) {
        N = atoi(argv[optind]);
        K = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = K = atoi(argv[optind]);
    }

    if (all) {
        fails += left(N, K, 0, verbose, &cf);
        fails += left(N, K, ARMAS_UNIT, verbose, &cf);
        fails += right(N, K, 0, verbose, &cf);
        fails += right(N, K, ARMAS_UNIT, verbose, &cf);
        fails += test_left_right(N, verbose, 0);
        fails += test_left_right(N, verbose, ARMAS_UNIT);
    } else {
        if (do_left)
            fails += left(N, K, unit, verbose, &cf);
        if (do_right)
            fails += right(N, K, unit, verbose, &cf);
        if (do_left && do_right)
            fails += test_left_right(N, verbose, unit);
    }

    exit(fails);
}
