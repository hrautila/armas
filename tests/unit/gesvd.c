
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-14
#endif

#define NAME "gesvd"

// test1: M > N and A, U are M-by-N, V = N-by-N
int test1(int M, int N, int verbose)
{
    armas_x_dense_t A, D, U, V, V1, W, A0, sD;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1, n2, nrm_A;
    long nwrk;
    int err, ok;

    if (M < N) {
        printf("%s [M=%d, N=%d]: test1 requires M >= N\n", PASS(0), M, N);
        return 0;
    }
    armas_x_init(&A, M, N);
    armas_x_init(&A0, M, N);
    armas_x_init(&U, M, N);
    armas_x_init(&V, N, N);
    armas_x_init(&V1, N, N);
    armas_x_init(&D, N, 1);
    
    armas_x_set_values(&A, unitrand, ARMAS_ANY);
    armas_x_mcopy(&A0, &A);
    nrm_A = armas_x_mnorm(&A, ARMAS_NORM_ONE, &conf);

    nwrk = armas_x_svd_work(&A, ARMAS_WANTU|ARMAS_WANTV, &conf);
    armas_x_init(&W, nwrk, 1);
    
    err = armas_x_svd(&D, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        printf("%s [M=%d, N=%d, err=%d,%d]: A == U*S*V.T, U=[m,n], V=[n,n] \n",
               PASS(0), M, N, err, conf.error);
        printf("Work needed: %d\n", armas_x_svd_work(&A, ARMAS_WANTU|ARMAS_WANTV, &conf));
        return 0;
    }

    armas_x_diag(&sD, &V1, 0);
    // compute: I - U.T*U
    armas_x_mult(0.0, &V1, 1.0, &U, &U, ARMAS_TRANSA, &conf);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_x_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: I - V*V.T
    armas_x_mult(0.0, &V1, 1.0, &V, &V, ARMAS_TRANSA, &conf);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_x_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T
    armas_x_diag(&sD, &A0, 0);
    armas_x_mult_diag(&U, &D, 1.0, ARMAS_RIGHT, &conf);
    armas_x_mult(1.0, &A0, -1.0, &U, &V, ARMAS_NONE, &conf);
    n2 = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf) / nrm_A;

    ok = isOK(n2, 10*M);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,n], V=[n,n] \n", PASS(ok), M, N);

    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%d]\n", n2, ndigits(n2));
        printf("  ||I - U.T*U||_1  : %e [%d]\n", n0, ndigits(n0));
        printf("  ||I - V*V.T||_1  : %e [%d]\n", n1, ndigits(n1));
    }

    armas_x_release(&A);
    armas_x_release(&A0);
    armas_x_release(&U);
    armas_x_release(&V);
    armas_x_release(&V1);
    armas_x_release(&D);
    armas_x_release(&W);
    return ok;
}

// test2: M > N and A, U are M-by-M, V = N-by-N
int test2(int M, int N, int verbose)
{
    armas_x_dense_t A, S, Sg, U, U1, V, V1, W, A0, A1, sD;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1, n2, nrm_A;
    int err, ok;

    if (M < N) {
        printf("%s [M=%d, N=%d]: test2 requires M >= N\n", PASS(0), M, N);
        return 0;
    }
    armas_x_init(&A, M, N);
    armas_x_init(&A0, M, N);
    armas_x_init(&A1, M, N);
    armas_x_init(&U, M, M);
    armas_x_init(&U1, M, M);
    armas_x_init(&V, N, N);
    armas_x_init(&V1, N, N);
    armas_x_init(&S, N, 1);
    armas_x_init(&Sg, M, N);
    
    armas_x_set_values(&A, unitrand, ARMAS_ANY);
    armas_x_set_values(&Sg, zero, ARMAS_ANY);
    armas_x_mcopy(&A0, &A);
    nrm_A = armas_x_mnorm(&A, ARMAS_NORM_ONE, &conf);

    armas_x_init(&W, (M*N < 100 ? 100 : M*N), 1);
    
    err = armas_x_svd(&S, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        //printf("error[%d]: %d\n", err, conf.error);
        printf("%s [M=%d, N=%d, err=%d,%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n",
               PASS(0), M, N, err, conf.error);
        return 0;
    }

    // compute: I - U.T*U
    armas_x_mult(0.0, &U1, 1.0, &U, &U, ARMAS_TRANSA, &conf);
    armas_x_diag(&sD, &U1, 0);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_x_mnorm(&U1, ARMAS_NORM_ONE, &conf);
    
    // compute: I - V*V.T
    armas_x_mult(0.0, &V1, 1.0, &V, &V, ARMAS_TRANSA, &conf);
    armas_x_diag(&sD, &V1, 0);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_x_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T == A - U*diag(Sg)*V.T
    armas_x_diag(&sD, &Sg, 0);
    armas_x_copy(&sD, &S, &conf);

    armas_x_mult(0.0, &A1, 1.0, &U, &Sg, ARMAS_NONE, &conf);
    armas_x_mult(1.0, &A0, -1.0, &A1, &V, ARMAS_NONE, &conf);
    n2 = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf) / nrm_A;

    ok = isOK(n2, 10*M);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n", PASS(ok), M, N);
    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%d]\n", n2, ndigits(n2));
        printf("  ||I - U.T*U||_1  : %e [%d]\n", n0, ndigits(n0));
        printf("  ||I - V*V.T||_1  : %e [%d]\n", n1, ndigits(n1));
    }

    armas_x_release(&A);
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&U);
    armas_x_release(&U1);
    armas_x_release(&V);
    armas_x_release(&V1);
    armas_x_release(&S);
    armas_x_release(&Sg);
    armas_x_release(&W);
    return ok;
}


// test3: M < N and A, U are M-by-M, V = M-by-N
int test3(int M, int N, int verbose)
{
    armas_x_dense_t A, D, U, U1, V, V1, W, A0, sD;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1, n2, nrm_A;
    int err, ok;

    if (M >= N) {
        printf("%s [M=%d, N=%d]: test3 requires M < N\n", PASS(0), M, N);
        return 0;
    }

    armas_x_init(&A, M, N);
    armas_x_init(&A0, M, N);
    armas_x_init(&U, M, M);
    armas_x_init(&U1, M, M);
    armas_x_init(&V, M, N);
    armas_x_init(&V1, N, N);
    armas_x_init(&D, M, 1);
    
    armas_x_set_values(&A, unitrand, ARMAS_ANY);
    armas_x_mcopy(&A0, &A);
    nrm_A = armas_x_mnorm(&A, ARMAS_NORM_ONE, &conf);

    armas_x_init(&W, (M*N < 100 ? 100 : N*M), 1);
    
    err = armas_x_svd(&D, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        printf("%s [M=%d, N=%d]: test3 err=%d,%d\n", PASS(0), M, N, err, conf.error);
        return 0;
    }

    armas_x_diag(&sD, &U1, 0);
    // compute: I - U.T*U
    armas_x_mult(0.0, &U1, 1.0, &U, &U, ARMAS_TRANSA, &conf);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_x_mnorm(&U1, ARMAS_NORM_ONE, &conf);

    //armas_x_diag(&sD, &V1, 0);
    // compute: I - V*V.T
    armas_x_mult(0.0, &U1, 1.0, &V, &V, ARMAS_TRANSB, &conf);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_x_mnorm(&U1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T
    armas_x_diag(&sD, &A0, 0);
    armas_x_mult_diag(&V, &D, 1.0, ARMAS_LEFT, &conf);
    armas_x_mult(1.0, &A0, -1.0, &U, &V, ARMAS_NONE, &conf);
    n2 = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf) / nrm_A;

    ok = isOK(n2, 10*N);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,m], V=[m,n] \n", PASS(ok), M, N);
    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%d]\n", n2, ndigits(n2));
        printf("  ||I - U.T*U||_1  : %e [%d]\n", n0, ndigits(n0));
        printf("  ||I - V*V.T||_1  : %e [%d]\n", n1, ndigits(n1));
    }

    armas_x_release(&A);
    armas_x_release(&A0);
    armas_x_release(&U);
    armas_x_release(&V);
    armas_x_release(&V1);
    armas_x_release(&U1);
    armas_x_release(&D);
    armas_x_release(&W);
    return ok;
}

// test4: M < N and A is M-by-N, U is M-by-M, V is N-by-N
int test4(int M, int N, int verbose)
{
    armas_x_dense_t A, S, Sg, U, U1, V, V1, W, A0, A1, sD;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1, n2, nrm_A;
    int err, ok;

    if (M > N) {
        printf("%s [M=%d, N=%d]: test4 requires M < N\n", PASS(0), M, N);
        return 0;
    }
    armas_x_init(&A, M, N);
    armas_x_init(&A0, M, N);
    armas_x_init(&A1, M, N);
    armas_x_init(&U, M, M);
    armas_x_init(&U1, M, M);
    armas_x_init(&V, N, N);
    armas_x_init(&V1, N, N);
    armas_x_init(&S, M, 1);
    armas_x_init(&Sg, M, N);
    
    armas_x_set_values(&A, unitrand, ARMAS_ANY);
    nrm_A = armas_x_mnorm(&A, ARMAS_NORM_ONE, &conf);

    armas_x_set_values(&Sg, zero, ARMAS_ANY);
    armas_x_mcopy(&A0, &A);
    armas_x_init(&W, (M*N < 100 ? 100 : M*N), 1);
    
    err = armas_x_svd(&S, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        printf("%s [M=%d, N=%d, err=%d,%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n",
               PASS(0), M, N, err, conf.error);
        return 0;
    }

    // compute: I - U.T*U
    armas_x_mult(0.0, &U1, 1.0, &U, &U, ARMAS_TRANSA, &conf);
    armas_x_diag(&sD, &U1, 0);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_x_mnorm(&U1, ARMAS_NORM_ONE, &conf);
    
    // compute: I - V*V.T
    armas_x_mult(0.0, &V1, 1.0, &V, &V, ARMAS_TRANSA, &conf);
    armas_x_diag(&sD, &V1, 0);
    armas_x_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_x_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T == A - U*diag(Sg)*V.T
    armas_x_diag(&sD, &Sg, 0);
    armas_x_copy(&sD, &S, &conf);

    armas_x_mult(0.0, &A1, 1.0, &Sg, &V, ARMAS_NONE, &conf);
    armas_x_mult(1.0, &A0, -1.0, &U, &A1, ARMAS_NONE, &conf);
    n2 = armas_x_mnorm(&A0, ARMAS_NORM_ONE, &conf) / nrm_A;

    ok = isOK(n2, 10*N);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n", PASS(ok), M, N);
    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%d]\n", n2, ndigits(n2));
        printf("  ||I - U.T*U||_1  : %e [%d]\n", n0, ndigits(n0));
        printf("  ||I - V*V.T||_1  : %e [%d]\n", n1, ndigits(n1));
    }

    armas_x_release(&A);
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&U);
    armas_x_release(&U1);
    armas_x_release(&V);
    armas_x_release(&V1);
    armas_x_release(&S);
    armas_x_release(&Sg);
    armas_x_release(&W);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 313;
    int N = 277;
    int Mbig = M;
    int Nsmall = Mbig/3;
    int LB = 36;
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
    
    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        M = atoi(argv[optind]);
        N = M;
    }

    if (M < N) {
        LB = M; M = N; N = LB;
    }
    if (M/3 < 10) {
        Nsmall = N;
        Mbig = 3*M;
    } else {
        Mbig = M;
        Nsmall = M/3;
    }

    int fails = 0;

    printf("Test: M >= N\n");
    if (! test1(M, N, verbose))
        fails++;
    if (! test2(M, N, verbose))
        fails++;

    printf("Test: M >> N\n");
    if (! test1(Mbig, Nsmall, verbose))
        fails++;
    if (! test2(Mbig, Nsmall, verbose))
        fails++;

    if (M != N) {
        // these only if M, N not equal
        printf("Test: M < N\n");
        if (! test3(N, M, verbose))
            fails++;
        if (! test4(N, M, verbose))
            fails++;

        printf("Test: M << N\n");
        if (! test3(Nsmall, Mbig, verbose))
            fails++;
        if (! test4(Nsmall, Mbig, verbose))
            fails++;
    }
    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
