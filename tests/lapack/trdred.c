
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define ERROR 6e-6
#else
#define ERROR 1e-12
#endif

#ifndef ARMAS_NIL
#define ARMAS_NIL (armas_x_dense_t *)0
#endif

#define NAME "trdreduce"

int test_reduce(int M, int N, int lb, int verbose, int flags)
{
    armas_x_dense_t A0, A1, tau0, tau1;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    char uplo = (flags & ARMAS_LOWER) ? 'L' : 'U';
    DTYPE n0, n1;

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&tau0, N, 1);
    armas_x_init(&tau1, N, 1);

    // set source data
    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0, 0, &conf);
    if (verbose > 1) {
        MAT_PRINT("Initial A", &A0);
    }
    env->lb = 0;
    armas_x_trdreduce(&A0, &tau0, flags, &conf);
    if (verbose > 1) {
        MAT_PRINT("unblk trd(A)", &A0);
    }

    env->lb = lb;
    armas_x_trdreduce(&A1, &tau1, flags, &conf);
    if (verbose > 1) {
        MAT_PRINT("blk trd(A)", &A1);
    }

    n0 = rel_error((DTYPE *) 0, &A0, &A1, ARMAS_NORM_ONE, 0, &conf);
    n1 = rel_error((DTYPE *) 0, &tau0, &tau1, ARMAS_NORM_TWO, 0, &conf);
    ok = n0 == ABSZERO || isFINE(n0, N * ERROR);

    printf("%s: unblk.TRD(A,%c) == blk.TRD(A,%c)\n", PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("  || error.TRD(A,%c)||: %e [%d]\n", uplo, n0, ndigits(n0));
        printf("  || error.tau||      : %e [%d]\n", n1, ndigits(n1));
    }
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&tau0);
    armas_x_release(&tau1);
    return ok;
}

// compute ||A - Q*T*Q^T||
int test_mult_trd(int M, int N, int lb, int verbose, int flags)
{
    armas_x_dense_t A0, A1, tau0, T0, T1, e1, e2, d1, d2;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    DTYPE nrm;
    const char *uplo = (flags & ARMAS_UPPER) ? "UPPER" : "LOWER";

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&T0, N, N);
    armas_x_init(&T1, N, N);
    armas_x_init(&tau0, N, 1);

    // set source data
    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0, 0, &conf);
    if (verbose > 1) {
        MAT_PRINT("A", &A0);
    }

    env->lb = lb;
    armas_x_trdreduce(&A0, &tau0, flags, &conf);

    // make tridiagonal matrix T0
    armas_x_diag(&d1, &A0, 0);
    armas_x_diag(&d2, &T0, 0);
    armas_x_mcopy(&d2, &d1, 0, &conf);
    if (flags & ARMAS_UPPER) {
        armas_x_diag(&e1, &A0, 1);
    } else {
        armas_x_diag(&e1, &A0, -1);
    }
    // copy off-diagonals
    armas_x_diag(&e2, &T0, 1);
    armas_x_mcopy(&e2, &e1, 0, &conf);
    armas_x_diag(&e2, &T0, -1);
    armas_x_mcopy(&e2, &e1, 0, &conf);
    if (verbose > 1) {
        MAT_PRINT("T", &T0);
    }
    // compute Q*T*Q^T
    armas_x_trdmult(&T0, &A0, &tau0, flags | ARMAS_LEFT, &conf);
    if (verbose > 1) {
        MAT_PRINT("Q*T", &T0);
    }
    armas_x_trdmult(&T0, &A0, &tau0, flags | ARMAS_RIGHT | ARMAS_TRANS, &conf);
    // make result triangular (original matrix)
    armas_x_make_trm(&T0, flags);
    if (verbose > 1) {
        MAT_PRINT("Q*T*Q^T", &T0);
    }

    nrm = rel_error((DTYPE *) 0, &T0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isFINE(nrm, N * ERROR);
    printf("%s: [%s] Q*T*Q.T == A\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&tau0);
    armas_x_release(&T0);
    armas_x_release(&T1);
    return ok;
}

// compute ||T - Q^T*A*Q||
int test_mult_a(int M, int N, int lb, int verbose, int flags)
{
    armas_x_dense_t A0, A1, tau0, T0, T1, e1, e2, d1, d2;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    DTYPE nrm;
    const char *uplo = (flags & ARMAS_UPPER) ? "UPPER" : "LOWER";

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&T0, N, N);
    armas_x_init(&T1, N, N);
    armas_x_init(&tau0, N, 1);

    // set source data; make it full symmetric
    armas_x_set_values(&A0, unitrand, ARMAS_SYMM);
    armas_x_mcopy(&A1, &A0, 0, &conf);

    env->lb = lb;
    armas_x_trdreduce(&A0, &tau0, flags, &conf);

    // make tridiagonal matrix T0
    armas_x_diag(&d1, &A0, 0);
    armas_x_diag(&d2, &T0, 0);
    armas_x_mcopy(&d2, &d1, 0, &conf);
    if (flags & ARMAS_UPPER) {
        armas_x_diag(&e1, &A0, 1);
    } else {
        armas_x_diag(&e1, &A0, -1);
    }
    // copy off-diagonals
    armas_x_diag(&e2, &T0, 1);
    armas_x_mcopy(&e2, &e1, 0, &conf);
    armas_x_diag(&e2, &T0, -1);
    armas_x_mcopy(&e2, &e1, 0, &conf);

    // compute Q.T*A*Q
    armas_x_trdmult(&A1, &A0, &tau0, flags | ARMAS_LEFT | ARMAS_TRANS, &conf);
    armas_x_trdmult(&A1, &A0, &tau0, flags | ARMAS_RIGHT, &conf);

    nrm = rel_error((DTYPE *) 0, &T0, &A1, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = isFINE(nrm, N * ERROR);
    printf("%s: [%s] Q.T*A*Q == T\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&tau0);
    armas_x_release(&T0);
    armas_x_release(&T1);
    return ok;
}

int test_build(int M, int N, int lb, int K, int verbose, int flags)
{
    armas_x_dense_t A0, tauq0, d0, QQt;
    armas_conf_t conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    int ok;
    DTYPE nrm;
    int tN = M < N ? M : N;
    const char *uplo = (flags & ARMAS_UPPER) ? "UPPER" : "LOWER";

    armas_x_init(&A0, N, N);
    armas_x_init(&tauq0, tN, 1);
    //------------------------------------------------------

    // set source data
    armas_x_set_values(&A0, unitrand, flags);
    // reduce to tridiagonal matrix
    env->lb = lb;
    armas_x_trdreduce(&A0, &tauq0, flags, &conf);
    // ----------------------------------------------------------------
    // Q-matrix
    armas_x_trdbuild(&A0, &tauq0, K, flags, &conf);
    armas_x_init(&QQt, N, N);
    armas_x_mult(ZERO, &QQt, ONE, &A0, &A0, ARMAS_TRANSB, &conf);
    armas_x_diag(&d0, &QQt, 0);
    armas_x_madd(&d0, -ONE, 0, &conf);

    nrm = armas_x_mnorm(&QQt, ARMAS_NORM_ONE, &conf);

    ok = isFINE(nrm, N * ERROR);
    printf("%s: [%s] I == Q*Q.T\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", nrm, ndigits(nrm));
    }
    //------------------------------------------------------
    armas_x_release(&A0);
    armas_x_release(&tauq0);
    armas_x_release(&QQt);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 741;
    int LB = 64;
    int verbose = 1;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [M N LB]\n", NAME);
            exit(1);
        }
    }

    if (optind < argc - 1) {
        N = atoi(argv[optind]);
        LB = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        LB = 0;
    }

    int fails = 0;

    if (!test_reduce(N, N, LB, verbose, ARMAS_LOWER))
        fails++;
#if 0
    if (!test_reduce(N, N, LB, verbose, ARMAS_UPPER))
        fails++;
#endif
    if (!test_mult_trd(N, N, LB, verbose, ARMAS_LOWER))
        fails++;
#if 0
    if (!test_mult_trd(N, N, LB, verbose, ARMAS_UPPER))
        fails++;
    if (!test_mult_a(N, N, LB, verbose, ARMAS_LOWER))
        fails++;
    if (!test_mult_a(N, N, LB, verbose, ARMAS_UPPER))
        fails++;
    if (!test_build(N, N, LB, N / 2, verbose, ARMAS_LOWER))
        fails++;
    if (!test_build(N, N, LB, N / 2, verbose, ARMAS_UPPER))
        fails++;
#endif
    exit(fails);
}
