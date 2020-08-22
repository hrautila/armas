
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"

int test_left_ext(int N, int K, int unit, int verbose, armas_conf_t *cf)
{
    armas_d_dense_t Y, Y1, Y0, A, At, L, Lt, E, Et;
    DTYPE n0, n1;
    int ok, fails = 0;

    armas_d_init(&Y, N, K);
    armas_d_init(&Y0, N, K);
    armas_d_init(&Y1, N, K);
    armas_d_init(&E, N, K);
    armas_d_init(&Et, N, K);
    armas_d_init(&A, N, N);
    armas_d_init(&At, N, N);
    armas_d_init(&L, N, N);
    armas_d_init(&Lt, N, N);

    armas_d_set_values(&Y, one, ARMAS_NULL);
    armas_d_mcopy(&Y1, &Y, 0, cf);
    make_ext_trsm_matrix(N, ARMAS_LEFT, &A, &At, &E, &Et, cf);
    armas_d_mcopy(&Lt, &A, ARMAS_TRANS, cf);
    armas_d_mcopy(&L, &At, ARMAS_TRANS, cf);

    if (verbose > 2 && N < 10) {
        printf("U\n"); armas_d_printf(stdout, "%9.2e", &A);
        printf("E\n"); armas_d_printf(stdout, "%9.2e", &E);
        printf("Ut\n"); armas_d_printf(stdout, "%9.2e", &At);
        printf("Et\n"); armas_d_printf(stdout, "%9.2e", &Et);
        printf("L\n"); armas_d_printf(stdout, "%9.2e", &L);
        printf("Lt\n"); armas_d_printf(stdout, "%9.2e", &Lt);
    }

    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &E, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &A, ARMAS_UPPER, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);

    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trsm(X, LEFT|U)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &Et, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &At, ARMAS_UPPER|ARMAS_TRANS, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);
    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);

    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trsm(X, LEFT|U|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    // ---------------------------------------------------------
    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &Et, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &L, ARMAS_LOWER, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);
    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);

    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trmm(X, LEFT|L)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &E, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &Lt, ARMAS_LOWER|ARMAS_TRANS, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);
    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);

    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trmm(X, LEFT|L|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    return fails;
}

int test_right_ext(int N, int K, int unit, int verbose, armas_conf_t *cf)
{
    armas_d_dense_t Y0, A, At, L, Lt, E, Et;
    DTYPE n0, n1;
    int ok, fails = 0;

    armas_d_init(&Y0, K, N);
    armas_d_init(&E, K, N);
    armas_d_init(&Et, K, N);
    armas_d_init(&A, N, N);
    armas_d_init(&At, N, N);
    armas_d_init(&L, N, N);
    armas_d_init(&Lt, N, N);

    armas_d_set_values(&Y0, one, ARMAS_NULL);
    make_ext_trsm_matrix(N, ARMAS_RIGHT, &A, &At, &E, &Et, cf);
    armas_d_mcopy(&Lt, &A, ARMAS_TRANS, cf);
    armas_d_mcopy(&L, &At, ARMAS_TRANS, cf);

    if (verbose > 2 && N < 10) {
        printf("U\n"); armas_d_printf(stdout, "%9.2e", &A);
        printf("E\n"); armas_d_printf(stdout, "%9.2e", &E);
        printf("Ut\n"); armas_d_printf(stdout, "%9.2e", &At);
        printf("Et\n"); armas_d_printf(stdout, "%9.2e", &Et);
        printf("L\n"); armas_d_printf(stdout, "%9.2e", &L);
        printf("Lt\n"); armas_d_printf(stdout, "%9.2e", &Lt);
    }

    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &E, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &A, ARMAS_RIGHT|ARMAS_UPPER, cf);
    if (verbose > 2 && N < 10) {
        printf("Y*A^-1:\n"); armas_d_printf(stdout, "%9.2e", &Y0);
    }
    armas_x_madd(&Y0, 1.0, 0, cf);

    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trsm(X, RIGHT|U)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &Et, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &At, ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANS, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);
    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);

    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trsm(X, RIGHT|U|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    // ---------------------------------------------------------
    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &Et, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &L, ARMAS_RIGHT|ARMAS_LOWER, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);
    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);

    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trmm(X, RIGHT|L)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    // ---------------------------------------------------------
#if 1
    armas_d_mcopy(&Y0, &E, 0, cf);
    armas_x_ext_solve_trm(&Y0, -1.0, &Lt, ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANS, cf);
    armas_x_madd(&Y0, 1.0, 0, cf);
    n0 = rel_error(&n1, &Y0, __nil, ARMAS_NORM_INF, 0, cf);

    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : expected == ext_trmm(X, RIGHT|L|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
#endif
    return fails;
}

/*
 *
 */
int main(int argc, char **argv)
{

    armas_conf_t cf;

    int opt;
    int N = 33;
    int K = 33;
    int fails = 0;
    int verbose = 0;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr,
                    "usage: ext_trsm [-p bits -C cond -v -SLRT] [size]\n");
            exit(1);
        }
    }

    if (optind < argc - 1) {
        N = atoi(argv[optind]);
        K = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        K = N;
    }

    cf = *armas_conf_default();
    fails += test_left_ext(N, K, 0, verbose, &cf);
    fails += test_right_ext(N, K, 0, verbose, &cf);

    exit(fails);
}
