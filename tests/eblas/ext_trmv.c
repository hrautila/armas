
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"


/*
 * Create test matrix for computing A*1 = E:
 *
 *   N e e e -N   3e   -N -N -N -N -N   -N
 *   . N e e -N   2e    .  N  e  e  e    0
 *   . . N e -N    e    .  .  N  e  e    e
 *   . . . N -N    0    .  .  .  N  e   2e
 *   . . . . -N   -N    .  .  .  .  N   3e
 *
 */

#define C 1000.0


// Compute sum of N integers.
static inline
int sum_n(int n)
{
    if ((n & 0x1) != 0) {
        return n*(n-1)/2 + n;
    }
    return (1+n)*n/2;
}

int check_std_vs_ext(int *nc, const armas_dense_t *diff, int trans)
{
    armas_dense_t e1;
    int count = 0, N = armas_size(diff);
    armas_subvector_unsafe(&e1, diff, trans, N-1);

    // |std - ext| >= 0 for all entries
    for (int i = 0; i < N-1; ++i) {
        count += ABS(armas_get_at(&e1, i)) >= ZERO ? 1 : 0;
    }
    // last (first) difference is zero
    DTYPE d = armas_get_at(&e1, trans ? 0 : N-2);
    if (nc)
        *nc = count;
    return count == N-1 && d == ZERO;
}

int compute_stats(DTYPE *avg_err, armas_dense_t *std, const armas_dense_t *ext, int num_eps, armas_conf_t *cf)
{
    DTYPE n0;
    // compute: n0 = sum(|std - ext|)
    armas_axpy(std, -1.0, ext, cf);
    n0 = armas_asum(std, cf);

    *avg_err = ABS(n0 - num_eps*EPS)/num_eps;
    // number of eps needed for sum
    return (int)(n0/EPS);
}

int test_compare_to_std(int N, int verbose, int unit, armas_conf_t *cf)
{
    armas_dense_t Y0, Y1, Y, A, At, L, Lt, E, Et;
    DTYPE avg_err;
    int ok, nc;
    int fails = 0;

    armas_init(&Y, N, 1);
    armas_init(&Y0, N, 1);
    armas_init(&Y1, N, 1);
    armas_init(&E, N, 1);
    armas_init(&Et, N, 1);
    armas_init(&A, N, N);
    armas_init(&At, N, N);

    armas_set_values(&Y, one, ARMAS_NULL);
    armas_mcopy(&Y0, &Y, 0, cf);
    armas_mcopy(&Y1, &Y, 0, cf);
    make_ext_trmv_data(N, &A, &At, &E, &Et);

    printf("** compare to computation with standard precision\n");

    // -------------------------------------------------------------
    armas_ext_mvmult_trm(&Y, 1.0, &A, ARMAS_UPPER, cf);
    armas_mvmult_trm(&Y1, 1.0, &A, ARMAS_UPPER, cf);

    compute_stats(&avg_err, &Y1, &Y, sum_n(N-2), cf);
    ok = check_std_vs_ext(&nc, &Y1, 0);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, U) >= trmv(X, U)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || avg element error || : %e, [%d/%d]\n", avg_err, nc, N-1);
    }

    // -------------------------------------------------------------
    armas_mcopy(&Y, &Y0, 0, cf);
    armas_mcopy(&Y1, &Y0, 0, cf);

    armas_ext_mvmult_trm(&Y, 1.0, &At, ARMAS_UPPER|ARMAS_TRANS, cf);
    armas_mvmult_trm(&Y1, 1.0, &At, ARMAS_UPPER|ARMAS_TRANS, cf);

    compute_stats(&avg_err, &Y1, &Y, sum_n(N-2), cf);
    ok = check_std_vs_ext(&nc, &Y1, 1);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, U^T) >= trmv(X, U^T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || avg element error || : %e, [%d/%d]\n", avg_err, nc, N-1);
    }

    // -------------------------------------------------------------
    armas_init(&L, N, N);
    armas_init(&Lt, N, N);
    armas_mcopy(&L, &At, ARMAS_TRANS, cf);
    armas_mcopy(&Lt, &A, ARMAS_TRANS, cf);
    armas_mcopy(&Y, &Y0, 0, cf);
    armas_mcopy(&Y1, &Y0, 0, cf);
    // -------------------------------------------------------------

    armas_ext_mvmult_trm(&Y, 1.0, &L, ARMAS_LOWER, cf);
    armas_mvmult_trm(&Y1, 1.0, &L, ARMAS_LOWER, cf);

    compute_stats(&avg_err, &Y1, &Y, sum_n(N-2), cf);
    ok = check_std_vs_ext(&nc, &Y1, 1);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, L) >= trmv(X, L)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || avg element error || : %e, [%d/%d]\n", avg_err, nc, N-1);
    }

    armas_mcopy(&Y, &Y0, 0, cf);
    armas_mcopy(&Y1, &Y0, 0, cf);
    // -------------------------------------------------------------

    armas_ext_mvmult_trm(&Y, 1.0, &Lt, ARMAS_LOWER|ARMAS_TRANS, cf);
    armas_mvmult_trm(&Y1, 1.0, &Lt, ARMAS_LOWER|ARMAS_TRANS, cf);

    compute_stats(&avg_err, &Y1, &Y, sum_n(N-2), cf);
    ok = check_std_vs_ext(&nc, &Y1, 0);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, L^T) >= trmv(X, L^T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || avg element error || : %e, [%d/%d]\n", avg_err, nc, N-1);
    }

    armas_release(&L);
    armas_release(&Lt);
    armas_release(&A);
    armas_release(&At);
    armas_release(&Y);
    armas_release(&Y0);
    armas_release(&Y1);
    armas_release(&E);
    armas_release(&Et);

    return fails;
}

int test_ext(int N, int verbose, int unit, armas_conf_t *cf)
{
    armas_dense_t Y0, Y1, Y, A, At, L, Lt, E, Et;
    DTYPE n0, n1;
    int ok;
    int fails = 0;

    armas_init(&Y, N, 1);
    armas_init(&Y0, N, 1);
    armas_init(&Y1, N, 1);
    armas_init(&E, N, 1);
    armas_init(&Et, N, 1);
    armas_init(&A, N, N);
    armas_init(&At, N, N);

    armas_set_values(&Y, one, ARMAS_NULL);
    armas_mcopy(&Y0, &Y, 0, cf);
    armas_mcopy(&Y1, &Y, 0, cf);
    make_ext_trmv_data(N, &A, &At, &E, &Et);

    printf("** computation with extended precision\n");

    if (verbose > 2) {
        armas_dense_t t;
        MAT_PRINT("A", &A);
        MAT_PRINT("A^T", &At);
        MAT_PRINT("E", armas_col_as_row(&t, &E));
        MAT_PRINT("E^T", armas_col_as_row(&t, &Et));
    }

    // -------------------------------------------------------------
    armas_ext_mvmult_trm(&Y, 1.0, &A, ARMAS_UPPER, cf);
    armas_mvmult_trm(&Y1, 1.0, &A, ARMAS_UPPER, cf);
    n0 = rel_error(&n1, &Y, &E, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, U) == expected\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    // -------------------------------------------------------------
    armas_mcopy(&Y, &Y0, 0, cf);
    armas_mcopy(&Y1, &Y0, 0, cf);

    armas_ext_mvmult_trm(&Y, 1.0, &At, ARMAS_UPPER|ARMAS_TRANS, cf);
    armas_mvmult_trm(&Y1, 1.0, &At, ARMAS_UPPER|ARMAS_TRANS, cf);
    n0 = rel_error(&n1, &Y, &Et, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, U^T) == expected\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // -------------------------------------------------------------
    armas_init(&L, N, N);
    armas_init(&Lt, N, N);
    armas_mcopy(&L, &At, ARMAS_TRANS, cf);
    armas_mcopy(&Lt, &A, ARMAS_TRANS, cf);

    armas_mcopy(&Y, &Y0, 0, cf);
    armas_mcopy(&Y1, &Y0, 0, cf);
    // -------------------------------------------------------------

    armas_ext_mvmult_trm(&Y, 1.0, &L, ARMAS_LOWER, cf);
    armas_mvmult_trm(&Y1, 1.0, &L, ARMAS_LOWER, cf);
    n0 = rel_error(&n1, &Y, &Et, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, L) == expected\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_mcopy(&Y, &Y0, 0, cf);
    armas_mcopy(&Y1, &Y0, 0, cf);
    // -------------------------------------------------------------

    armas_ext_mvmult_trm(&Y, 1.0, &Lt, ARMAS_LOWER|ARMAS_TRANS, cf);
    armas_mvmult_trm(&Y1, 1.0, &Lt, ARMAS_LOWER|ARMAS_TRANS, cf);
    n0 = rel_error(&n1, &Y, &E, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N);
    fails += (1 - ok);
    printf("%6s : ext_trmv(X, L^T) == expected\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_release(&L);
    armas_release(&Lt);
    armas_release(&A);
    armas_release(&At);
    armas_release(&Y);
    armas_release(&Y0);
    armas_release(&Y1);
    armas_release(&E);
    armas_release(&Et);

    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;
    int opt;
    int N = 911;
    int fails = 0;
    int verbose = 0;
    int unit = 0;
    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vr:ub:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'b':
            env->blas1min = atoi(optarg);
            break;
        case 'u':
            unit = ARMAS_UNIT;
            break;
        default:
            fprintf(stderr, "usage: trmv [size]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    fails += test_ext(N, verbose, unit, &cf);
    fails += test_compare_to_std(N, verbose, unit, &cf);
    exit(fails);
}
