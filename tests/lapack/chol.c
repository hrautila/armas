
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#if FLOAT32
#define __ERROR 8e-3
#else
#define __ERROR 1e-8
#endif

#define NAME "chol"

static
int test_solve(int M, int N, int lb, int verbose, int flags)
{
    armas_dense_t A0, A1;
    armas_dense_t B0, X0;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    DTYPE nrm, nrm0;
    const char *uplo = (flags & ARMAS_UPPER) ? "Upper" : "Lower";
    const char *blk = lb != 0 ? "  blk" : "unblk";

    armas_init(&A0, N, N);
    armas_init(&A1, N, N);
    armas_init(&B0, N, M);
    armas_init(&X0, N, M);

    // set source data (A = A^T*A)
    armas_set_values(&A0, zeromean, ARMAS_ANY);
    armas_mult(0.0, &A1, 1.0, &A0, &A0, ARMAS_TRANSA, &conf);
    armas_mcopy(&A0, &A1, 0, &conf);

    armas_set_values(&B0, unitrand, ARMAS_ANY);
    nrm0 = armas_mnorm(&B0, ARMAS_NORM_ONE, &conf);
    armas_mcopy(&X0, &B0, 0, &conf);

    env->lb = lb;
    armas_cholfactor(&A0, ARMAS_NOPIVOT, flags, &conf);

    // solve
    armas_cholsolve(&X0, &A0, ARMAS_NOPIVOT, flags, &conf);

    // X0 = A*X0 - B0
    armas_mult(1.0, &B0, -1.0, &A1, &X0, ARMAS_NONE, &conf);
    nrm = armas_mnorm(&B0, ARMAS_NORM_ONE, &conf) / nrm0;
    ok = isFINE(nrm, N * __ERROR);

    printf("%s: A*(%s.CHOLsolve(A, B, %s)) == B\n", PASS(ok), blk, uplo);
    if (verbose > 0) {
        printf("   || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&B0);
    armas_release(&X0);

    return ok;
}

static
int test_factor(int M, int N, int lb, int verbose, int flags)
{
    armas_dense_t A0, A1;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    DTYPE nrm;
    const char uplo = (flags & ARMAS_UPPER) ? 'U' : 'L';
    armas_init(&A0, N, N);
    armas_init(&A1, N, N);

    // set source data
    armas_set_values(&A0, unitrand, ARMAS_ANY);
    // A = A^T*A; positive semi-definite
    armas_mult(0.0, &A1, 1.0, &A0, &A0, ARMAS_TRANSA, &conf);
    armas_mcopy(&A0, &A1, 0, &conf);
    if (verbose > 1) {
        MAT_PRINT("A", &A0);
    }

    env->lb = 0;
    armas_cholfactor(&A0, ARMAS_NOPIVOT, flags, &conf);
    if (verbose > 1) {
        MAT_PRINT("A0", &A0);
    }
    env->lb = lb;
    armas_cholfactor(&A1, ARMAS_NOPIVOT, flags, &conf);
    if (verbose > 1) {
        MAT_PRINT("A1", &A1);
    }

    nrm = rel_error((DTYPE *) 0, &A0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isOK(nrm, N);
    if (verbose > 1) {
        MAT_PRINT("A0 - A1", &A0);
    }

    printf("%s: unblk.CHOL(A,%c) == blk.CHOL(A,%c)\n", PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("   || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }

    armas_release(&A0);
    armas_release(&A1);
    return ok;
}

static
int test_cholpv(int N, int lb, int verbose, int flags)
{
    armas_dense_t A0, A1, C, D, W;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    armas_pivot_t P;
    DTYPE n0, n1;
    int ok, flags1, flags2, pflgs;
    const char *fact = (flags & ARMAS_LOWER) ? "P^T*(LL^T)*P" : "P^T*(U^TU)*P";
    const char *blk = (lb == 0) ? "unblk" : "  blk";
    pflgs = (flags & ARMAS_LOWER) ? ARMAS_PIVOT_LOWER : ARMAS_PIVOT_UPPER;

    armas_init(&A0, N, N);
    armas_init(&A1, N, N);
    armas_init(&C, N, N);
    armas_init(&W, N, 1);
    armas_diag(&D, &C, 0);
    armas_madd(&D, 1.0, 0, &conf);
    armas_pivot_init(&P, N);

    armas_set_values(&A0, unitrand, 0);
    armas_mult(0.0, &A1, 1.0, &A0, &A0, ARMAS_TRANSA, &conf);
    armas_make_trm(&A1, flags);
    armas_mcopy(&A0, &A1, 0, &conf);

    env->lb = lb;
    armas_cholfactor(&A0, &P, flags, &conf);

    if (flags & ARMAS_LOWER) {
        flags1 = ARMAS_LOWER | ARMAS_RIGHT;
        flags2 = ARMAS_TRANS | ARMAS_LOWER | ARMAS_RIGHT;
    } else {
        flags2 = ARMAS_UPPER | ARMAS_RIGHT;
        flags1 = ARMAS_UPPER | ARMAS_TRANS | ARMAS_RIGHT;
    }

    // C = I*L*L.T || I*U.T*U
    armas_mult_trm(&C, 1.0, &A0, flags1, &conf);
    armas_mult_trm(&C, 1.0, &A0, flags2, &conf);
    armas_make_trm(&C, flags);

    armas_pivot(&C, &P, pflgs | ARMAS_PIVOT_BACKWARD, &conf);

    n0 = rel_error(&n1, &C, &A1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * __ERROR);
    printf("%s : %s.%s = A\n", PASS(ok), blk, fact);
    if (verbose > 0)
        printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&C);
    armas_release(&W);
    armas_pivot_release(&P);
    return ok;
}

static
int test_cholpv_solve(int M, int N, int lb, int verbose, int flags)
{
    armas_dense_t A0, A1, B, B0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    armas_pivot_t P0;
    DTYPE n0, n1;
    int e, ok;
    const char *fact = (flags & ARMAS_LOWER) != 0 ? "LL^T" : "U^TU";
    const char *blk = (lb == 0) ? "unblk" : "  blk";

    armas_init(&A0, N, N);
    armas_init(&A1, N, N);
    armas_init(&B0, N, M);
    armas_init(&B, N, M);
    armas_init(&W, N, 1);
    armas_pivot_init(&P0, N);

    armas_set_values(&A0, unitrand, 0);
    armas_mult(0.0, &A1, 1.0, &A0, &A0, ARMAS_TRANSA, &conf);
    armas_mcopy(&A0, &A1, 0, &conf);
    armas_make_trm(&A0, flags);

    armas_set_values(&B0, zeromean, 0);
    // B = A*B0
    armas_mult(0.0, &B, 1.0, &A1, &B0, 0, &conf);

    env->lb = lb;
    armas_cholfactor(&A0, &P0, flags, &conf);

    if ((e = armas_cholsolve(&B, &A0, &P0, flags, &conf)) < 0)
        printf("Error: solver error %d, [%d]\n", conf.error, e);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N * __ERROR);
    printf("%s : %s.(%s)^-1*B = X\n", PASS(ok), blk, fact);
    if (verbose > 0)
        printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_release(&A0);
    armas_release(&A1);
    armas_release(&B);
    armas_release(&B0);
    armas_pivot_release(&P0);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 511;
    int N = 779;
    int LB = 36;
    int verbose = 1;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v]  [M N LB]\n", NAME);
            exit(1);
        }
    }

    if (optind < argc - 2) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
        LB = atoi(argv[optind + 2]);
    } else if (optind < argc - 1) {
        N = atoi(argv[optind]);
        M = N;
        LB = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        M = N;
        LB = 0;
    }

    int fails = 0;

    if (!test_factor(M, N, LB, verbose, ARMAS_LOWER))
        fails++;
    if (!test_factor(M, N, LB, verbose, ARMAS_UPPER))
        fails++;

    if (!test_solve(M, N, LB, verbose, ARMAS_LOWER))
        fails++;

    if (!test_solve(M, N, LB, verbose, ARMAS_UPPER))
        fails++;

    if (!test_solve(M, N, 0, verbose, ARMAS_LOWER))
        fails++;

    if (!test_solve(M, N, 0, verbose, ARMAS_UPPER))
        fails++;

    if (!test_cholpv(N, 0, verbose, ARMAS_LOWER))
        fails++;
    if (!test_cholpv(N, LB, verbose, ARMAS_LOWER))
        fails++;
    if (!test_cholpv(N, 0, verbose, ARMAS_UPPER))
        fails++;
    if (!test_cholpv(N, LB, verbose, ARMAS_UPPER))
        fails++;
    if (!test_cholpv_solve(M, N, 0, verbose, ARMAS_LOWER))
        fails++;
    if (!test_cholpv_solve(M, N, LB, verbose, ARMAS_LOWER))
        fails++;
    if (!test_cholpv_solve(M, N, 0, verbose, ARMAS_UPPER))
        fails++;
    if (!test_cholpv_solve(M, N, LB, verbose, ARMAS_UPPER))
        fails++;
    exit(fails);
}
