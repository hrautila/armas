
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <armas/dmatrix.h>
#include "helper.h"

#define NAME "gesvd"

// test1: M > N and A, U are M-by-N, V = N-by-N
int test1(int M, int N, int verbose)
{
    armas_d_dense_t A, D, S1, E, U, U1, V, V1, W, A0, A1, sD;
    armas_conf_t conf = *armas_conf_default();
    double n0, n1, n2;
    int err, ok;

    if (M < N) {
        printf("%s [M=%d, N=%d]: test1 requires M >= N\n", PASS(0), M, N);
        return 0;
    }
    armas_d_init(&A, M, N);
    armas_d_init(&A0, M, N);
    armas_d_init(&U, M, N);
    armas_d_init(&V, N, N);
    armas_d_init(&V1, N, N);
    armas_d_init(&D, N, 1);
    
    armas_d_set_values(&A, unitrand, ARMAS_ANY);
    armas_d_mcopy(&A0, &A);
    
    armas_d_init(&W, (M*N < 100 ? 100 : M*N), 1);
    
    err = armas_d_svd(&D, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        printf("%s [M=%d, N=%d, err=%d,%d]: A == U*S*V.T, U=[m,n], V=[n,n] \n",
               PASS(0), M, N, err, conf.error);
        printf("Work needed: %d\n", armas_d_svd_work(&A, ARMAS_WANTU|ARMAS_WANTV, &conf));
        return 0;
    }

    armas_d_diag(&sD, &V1, 0);
    // compute: I - U.T*U
    armas_d_mult(&V1, &U, &U, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_d_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: I - V*V.T
    armas_d_mult(&V1, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_d_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T
    armas_d_diag(&sD, &A0, 0);
    armas_d_mult_diag(&U, &D, ARMAS_RIGHT, &conf);
    armas_d_mult(&A0, &U, &V, -1.0, 1.0, ARMAS_NONE, &conf);
    n2 = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);

    ok = isOK(n2, 10*M);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,n], V=[n,n] \n", PASS(ok), M, N);

    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%ld]\n", n2, (long)(n2/DBL_EPSILON));
        printf("  ||I - U.T*U||_1  : %e [%ld]\n", n0, (long)(n0/DBL_EPSILON));
        printf("  ||I - V*V.T||_1  : %e [%ld]\n", n1, (long)(n1/DBL_EPSILON));
    }

    armas_d_release(&A);
    armas_d_release(&A0);
    armas_d_release(&U);
    armas_d_release(&V);
    armas_d_release(&V1);
    armas_d_release(&D);
    armas_d_release(&W);
    return ok;
}

// test2: M > N and A, U are M-by-M, V = N-by-N
int test2(int M, int N, int verbose)
{
    armas_d_dense_t A, S, Sg, E, U, U1, V, V1, W, A0, A1, sD;
    armas_conf_t conf = *armas_conf_default();
    double n0, n1, n2;
    int err, ok;

    if (M < N) {
        printf("%s [M=%d, N=%d]: test2 requires M >= N\n", PASS(0), M, N);
        return 0;
    }
    armas_d_init(&A, M, N);
    armas_d_init(&A0, M, N);
    armas_d_init(&A1, M, N);
    armas_d_init(&U, M, M);
    armas_d_init(&U1, M, M);
    armas_d_init(&V, N, N);
    armas_d_init(&V1, N, N);
    armas_d_init(&S, N, 1);
    armas_d_init(&Sg, M, N);
    
    armas_d_set_values(&A, unitrand, ARMAS_ANY);
    armas_d_set_values(&Sg, zero, ARMAS_ANY);
    armas_d_mcopy(&A0, &A);
    armas_d_init(&W, (M*N < 100 ? 100 : M*N), 1);
    
    err = armas_d_svd(&S, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        //printf("error[%d]: %d\n", err, conf.error);
        printf("%s [M=%d, N=%d, err=%d,%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n",
               PASS(0), M, N, err, conf.error);
        return 0;
    }

    // compute: I - U.T*U
    armas_d_mult(&U1, &U, &U, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_diag(&sD, &U1, 0);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_d_mnorm(&U1, ARMAS_NORM_ONE, &conf);
    
    // compute: I - V*V.T
    armas_d_mult(&V1, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_diag(&sD, &V1, 0);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_d_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T == A - U*diag(Sg)*V.T
    armas_d_diag(&sD, &Sg, 0);
    armas_d_copy(&sD, &S, &conf);

    armas_d_mult(&A1, &U, &Sg, 1.0, 0.0, ARMAS_NONE, &conf);
    armas_d_mult(&A0, &A1, &V, -1.0, 1.0, ARMAS_NONE, &conf);
    n2 = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);

    ok = isOK(n2, 10*M);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n", PASS(ok), M, N);
    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%ld]\n", n2, (long)(n2/DBL_EPSILON));
        printf("  ||I - U.T*U||_1  : %e [%ld]\n", n0, (long)(n0/DBL_EPSILON));
        printf("  ||I - V*V.T||_1  : %e [%ld]\n", n1, (long)(n1/DBL_EPSILON));
    }

    armas_d_release(&A);
    armas_d_release(&A0);
    armas_d_release(&A1);
    armas_d_release(&U);
    armas_d_release(&U1);
    armas_d_release(&V);
    armas_d_release(&V1);
    armas_d_release(&S);
    armas_d_release(&Sg);
    armas_d_release(&W);
}


// test3: M < N and A, U are M-by-M, V = M-by-N
int test3(int M, int N, int verbose)
{
    armas_d_dense_t A, D, S1, E, U, U1, V, V1, W, A0, A1, sD;
    armas_conf_t conf = *armas_conf_default();
    double n0, n1, n2;
    int err, ok;

    if (M >= N) {
        printf("%s [M=%d, N=%d]: test3 requires M < N\n", PASS(0), M, N);
        return 0;
    }

    armas_d_init(&A, M, N);
    armas_d_init(&A0, M, N);
    armas_d_init(&U, M, M);
    armas_d_init(&U1, M, M);
    armas_d_init(&V, M, N);
    armas_d_init(&V1, N, N);
    armas_d_init(&D, M, 1);
    
    armas_d_set_values(&A, unitrand, ARMAS_ANY);
    armas_d_mcopy(&A0, &A);
    armas_d_init(&W, (M*N < 100 ? 100 : N*M), 1);
    
    err = armas_d_svd(&D, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        printf("%s [M=%d, N=%d]: test3 err=%d,%d\n", PASS(0), M, N, err, conf.error);
        return 0;
    }

    armas_d_diag(&sD, &U1, 0);
    // compute: I - U.T*U
    armas_d_mult(&U1, &U, &U, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_d_mnorm(&U1, ARMAS_NORM_ONE, &conf);

    //armas_d_diag(&sD, &V1, 0);
    // compute: I - V*V.T
    armas_d_mult(&U1, &V, &V, 1.0, 0.0, ARMAS_TRANSB, &conf);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_d_mnorm(&U1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T
    armas_d_diag(&sD, &A0, 0);
    armas_d_mult_diag(&V, &D, ARMAS_LEFT, &conf);
    armas_d_mult(&A0, &U, &V, -1.0, 1.0, ARMAS_NONE, &conf);
    n2 = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);

    ok = isOK(n2, 10*N);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,m], V=[m,n] \n", PASS(ok), M, N);
    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%ld]\n", n2, (long)(n2/DBL_EPSILON));
        printf("  ||I - U.T*U||_1  : %e [%ld]\n", n0, (long)(n0/DBL_EPSILON));
        printf("  ||I - V*V.T||_1  : %e [%ld]\n", n1, (long)(n1/DBL_EPSILON));
    }

    armas_d_release(&A);
    armas_d_release(&A0);
    armas_d_release(&U);
    armas_d_release(&V);
    armas_d_release(&V1);
    armas_d_release(&U1);
    armas_d_release(&D);
    armas_d_release(&W);
    return ok;
}

// test4: M < N and A is M-by-N, U is M-by-M, V is N-by-N
int test4(int M, int N, int verbose)
{
    armas_d_dense_t A, S, Sg, E, U, U1, V, V1, W, A0, A1, sD;
    armas_conf_t conf = *armas_conf_default();
    double n0, n1, n2;
    int err, ok;

    if (M > N) {
        printf("%s [M=%d, N=%d]: test4 requires M < N\n", PASS(0), M, N);
        return 0;
    }
    armas_d_init(&A, M, N);
    armas_d_init(&A0, M, N);
    armas_d_init(&A1, M, N);
    armas_d_init(&U, M, M);
    armas_d_init(&U1, M, M);
    armas_d_init(&V, N, N);
    armas_d_init(&V1, N, N);
    armas_d_init(&S, M, 1);
    armas_d_init(&Sg, M, N);
    
    armas_d_set_values(&A, unitrand, ARMAS_ANY);
    armas_d_set_values(&Sg, zero, ARMAS_ANY);
    armas_d_mcopy(&A0, &A);
    armas_d_init(&W, (M*N < 100 ? 100 : M*N), 1);
    
    err = armas_d_svd(&S, &U, &V, &A, &W, ARMAS_WANTU|ARMAS_WANTV, &conf);
    if (err) {
        printf("%s [M=%d, N=%d, err=%d,%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n",
               PASS(0), M, N, err, conf.error);
        return 0;
    }

    // compute: I - U.T*U
    armas_d_mult(&U1, &U, &U, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_diag(&sD, &U1, 0);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n0 = armas_d_mnorm(&U1, ARMAS_NORM_ONE, &conf);
    
    // compute: I - V*V.T
    armas_d_mult(&V1, &V, &V, 1.0, 0.0, ARMAS_TRANSA, &conf);
    armas_d_diag(&sD, &V1, 0);
    armas_d_madd(&sD, -1.0, ARMAS_ANY);
    n1 = armas_d_mnorm(&V1, ARMAS_NORM_ONE, &conf);

    // compute: A - U*S*V.T == A - U*diag(Sg)*V.T
    armas_d_diag(&sD, &Sg, 0);
    armas_d_copy(&sD, &S, &conf);

    armas_d_mult(&A1, &Sg, &V, 1.0, 0.0, ARMAS_NONE, &conf);
    armas_d_mult(&A0, &U, &A1, -1.0, 1.0, ARMAS_NONE, &conf);
    n2 = armas_d_mnorm(&A0, ARMAS_NORM_ONE, &conf);

    ok = isOK(n2, 10*N);
    printf("%s [M=%d, N=%d]: A == U*S*V.T, U=[m,m], V=[n,n] \n", PASS(ok), M, N);
    if (verbose > 0) {
        printf("  ||A - U*S*V.T||_1: %e [%ld]\n", n2, (long)(n2/DBL_EPSILON));
        printf("  ||I - U.T*U||_1  : %e [%ld]\n", n0, (long)(n0/DBL_EPSILON));
        printf("  ||I - V*V.T||_1  : %e [%ld]\n", n1, (long)(n1/DBL_EPSILON));
    }

    armas_d_release(&A);
    armas_d_release(&A0);
    armas_d_release(&A1);
    armas_d_release(&U);
    armas_d_release(&U1);
    armas_d_release(&V);
    armas_d_release(&V1);
    armas_d_release(&S);
    armas_d_release(&Sg);
    armas_d_release(&W);
}

main(int argc, char **argv)
{
    int opt;
    int M = 313;
    int N = 277;
    int Mbig = M;
    int Nsmall = Mbig/3;
    int LB = 36;
    int ok = 0;
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
