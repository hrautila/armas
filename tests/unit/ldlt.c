/**
 * Non-pivoting LDL.T factoring
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "ldlnp"

#if FLOAT32
#define __ERROR 1e-4
#else
#define __ERROR 1e-9
#endif

int test_ldl(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, C, D, D0, W;
    armas_conf_t conf = *armas_conf_default();
    __Dtype n0, n1;
    int ok, flags1, flags2;
    char *fact = flags & ARMAS_LOWER ? "L*D*L.T" : "U*D*U.T";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&C, N, N);
    matrix_init(&W, lb == 0 ? N : lb*N, 1);
    matrix_diag(&D, &C, 0);
    matrix_madd(&D, 1.0, 0);

    matrix_set_values(&A0, unitrand, flags);
    matrix_mcopy(&A1, &A0);

    conf.lb = lb;
    matrix_ldlfactor(&A0, &W, ARMAS_NOPIVOT, flags, &conf);
    matrix_diag(&D0, &A0, 0);

    flags1 = flags|ARMAS_UNIT|ARMAS_RIGHT;
    flags2 = flags|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT;

    // C = I*A; C = C*D; C = C*L^T
    matrix_mult_trm(&C, &A0, 1.0, flags1, &conf);
    matrix_mult_diag(&C, &D0, flags|ARMAS_RIGHT, &conf);
    matrix_mult_trm(&C, &A0, 1.0, flags2, &conf);
    matrix_make_trm(&C, flags);

    n0 = rel_error(&n1, &C, &A1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.%s = A\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&C);
    matrix_release(&W);
    return 1 - ok;
}

int test_ldl_solve(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, B, B0, W;
    armas_conf_t conf = *armas_conf_default();
    __Dtype n0, n1;
    int ok;
    char *fact = flags & ARMAS_LOWER ? "L*D*L.T" : "U*D*U.T";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&B0, N, N);
    matrix_init(&B, N, N);
    matrix_init(&W, lb == 0 ? N : lb*N, 1);

    matrix_set_values(&A0, unitrand, flags);
    matrix_mcopy(&A1, &A0);

    matrix_set_values(&B0, unitrand, 0);
    // B = A*B0
    matrix_mult_sym(&B, &A0, &B0, 1.0, 0.0, flags|ARMAS_LEFT, &conf);

    conf.lb = lb;
    matrix_ldlfactor(&A0, &W, ARMAS_NOPIVOT, flags, &conf);
    matrix_ldlsolve(&B, &A0, ARMAS_NOPIVOT, flags, &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.(%s).-1*B = X\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&B);
    matrix_release(&B0);
    matrix_release(&W);
    return 1 - ok;
}

// with pivoting; verify A = L*D*L.T or A = U*D*U.T
int test_ldlpv(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, C, D, D0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P;
    __Dtype n0, n1;
    int ok, flags1, flags2;
    char *fact = flags & ARMAS_LOWER ? "P^T*(LDL^T)*P" : "P^T*(UDU^T)*P";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&C, N, N);
    matrix_init(&W, lb == 0 ? N : lb*N, 1);
    matrix_diag(&D, &C, 0);
    matrix_madd(&D, 1.0, 0);
    armas_pivot_init(&P, N);
    
    matrix_set_values(&A0, unitrand, flags);
    matrix_mcopy(&A1, &A0);
    if (N < 10) {
        printf("A:\n"); matrix_printf(stdout, "%6.3f", &A0);
    }

    conf.lb = lb;
    matrix_ldlfactor(&A0, &W, &P, flags, &conf);
    matrix_diag(&D0, &A0, 0);

    flags1 = flags|ARMAS_UNIT|ARMAS_RIGHT;
    flags2 = flags|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT;

    // C = ((I*L)*D)*L^T
    matrix_mult_trm(&C, &A0, 1.0, flags1, &conf);
    matrix_mult_diag(&C, &D0, flags|ARMAS_RIGHT, &conf);
    matrix_mult_trm(&C, &A0, 1.0, flags2, &conf);
    matrix_make_trm(&C, flags);
    if (flags & ARMAS_LOWER) {
        matrix_pivot(&C, &P, ARMAS_PIVOT_LOWER|ARMAS_PIVOT_BACKWARD, &conf);
    } else {
        matrix_pivot(&C, &P, ARMAS_PIVOT_UPPER|ARMAS_PIVOT_FORWARD, &conf);
    }

    if (N < 10) {
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P);
        printf("%s:\n", fact); matrix_printf(stdout, "%6.3f", &C);
    }
    n0 = rel_error(&n1, &C, &A1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.%s = A\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&C);
    matrix_release(&W);
    armas_pivot_release(&P);
    return 1 - ok;
}

// with pivoting; verify (L*D*L.T).T = U*D*U.T
int test_ldlpv_transpose(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, A2, C0, C1, D1, D0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P0, P1;
    __Dtype n0, n1;
    int ok;
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&A2, N, N);
    matrix_init(&C0, N, N);
    matrix_init(&C1, N, N);
    matrix_init(&W, lb == 0 ? N : lb*N, 1);
    matrix_diag(&D0, &C0, 0);
    matrix_madd(&D0, 1.0, 0);
    matrix_mcopy(&C1, &C0);
    matrix_diag(&D1, &C1, 0);
    armas_pivot_init(&P0, N);
    armas_pivot_init(&P1, N);
    
    matrix_set_values(&A0, unitrand, ARMAS_SYMM);
    matrix_mcopy(&A1, &A0);
    matrix_mcopy(&A2, &A0);
    matrix_make_trm(&A0, ARMAS_LOWER);
    matrix_make_trm(&A1, ARMAS_UPPER);
    if (N < 10) {
        printf("A:\n"); matrix_printf(stdout, "%6.3f", &A2);
    }

    conf.lb = lb;
    matrix_ldlfactor(&A0, &W, &P0, ARMAS_LOWER, &conf);
    matrix_diag(&D0, &A0, 0);
    matrix_ldlfactor(&A1, &W, &P1, ARMAS_UPPER, &conf);
    matrix_diag(&D1, &A1, 0);

    // C = ((I*L)*D)*L.T
    matrix_mult_trm(&C0, &A0, 1.0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    matrix_mult_diag(&C0, &D0, ARMAS_LOWER|ARMAS_RIGHT, &conf);
    matrix_mult_trm(&C0, &A0, 1.0, ARMAS_LOWER|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    matrix_make_trm(&C0, ARMAS_LOWER);
    matrix_pivot(&C0, &P0, ARMAS_PIVOT_LOWER|ARMAS_PIVOT_BACKWARD, &conf);

    if (N < 10) {
        printf("(1) P^T*(LDL^T)*P:\n"); matrix_printf(stdout, "%6.3f", &C0);
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P0);
    }
    // C = ((I*U)*D)*U.T
    matrix_mult_trm(&C1, &A1, 1.0, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    matrix_mult_diag(&C1, &D1, ARMAS_UPPER|ARMAS_RIGHT, &conf);
    matrix_mult_trm(&C1, &A1, 1.0, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    matrix_make_trm(&C1, ARMAS_UPPER);
    matrix_pivot(&C1, &P1, ARMAS_PIVOT_UPPER|ARMAS_PIVOT_FORWARD, &conf);

    if (N < 10) {
        printf("(2) P^T*(UDU^T)*P:\n"); matrix_printf(stdout, "%6.3f", &C1);
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P1);
    }

    n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_INF, ARMAS_TRANSB, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.P^T*(LDL^T)*P = transpose(%s.P^T*(UDU^T)*P)\n", PASS(ok), blk, blk);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&A2);
    matrix_release(&C0);
    matrix_release(&C1);
    matrix_release(&W);
    armas_pivot_release(&P0);
    armas_pivot_release(&P1);
    return 1 - ok;
}

int test_ldlpv_solve(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, B, B0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P0;
    __Dtype n0, n1;
    int ok;
    char *fact = flags & ARMAS_LOWER ? "P^T*(LDL^T)*P" : "P^T*(UDU^T)*P";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&B0, N, N);
    matrix_init(&B, N, N);
    matrix_init(&W, lb == 0 ? N : lb*N, 1);
    armas_pivot_init(&P0, N);
    
    matrix_set_values(&A0, unitrand, flags);
    matrix_mcopy(&A1, &A0);

    matrix_set_values(&B0, unitrand, 0);
    // B = A*B0
    matrix_mult_sym(&B, &A0, &B0, 1.0, 0.0, flags|ARMAS_LEFT, &conf);

    if (N < 10) {
        printf("B0:\n"); matrix_printf(stdout, "%6.3f", &B0);
    }

    conf.lb = lb;
    matrix_ldlfactor(&A0, &W, &P0, flags, &conf);
    matrix_ldlsolve(&B, &A0, &P0, flags, &conf);

    if (N < 10) {
        printf("(%s)^-1*B:\n", fact); matrix_printf(stdout, "%6.3f", &B);
    }
    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.(%s)^-1*B = X\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&B);
    matrix_release(&B0);
    matrix_release(&W);
    armas_pivot_release(&P0);
    return 1 - ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 151;
    int LB = 16;
    int verbose = 0;

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
        N = atoi(argv[optind]);
        LB = atoi(argv[optind+1]);
    } else if (optind < argc-1) {
        N = atoi(argv[optind]);
        LB = N > 32 ? 24 : 8;
    }

    int fails = 0;


    if (test_ldl(N, 0, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldl(N, 0, ARMAS_UPPER, verbose))
        fails++;
    if (test_ldl(N, LB, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldl(N, LB, ARMAS_UPPER, verbose))
        fails++;;

    if (test_ldlpv(N, 0, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldlpv(N, 0, ARMAS_UPPER, verbose))
        fails++;
    if (test_ldlpv(N, LB, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldlpv(N, LB, ARMAS_UPPER, verbose))
        fails++;;

    if (test_ldlpv_transpose(N, 0, 0, verbose))
        fails++;
    if (test_ldlpv_transpose(N, LB, 0, verbose))
        fails++;

    if (test_ldl_solve(N, 0, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldl_solve(N, 0, ARMAS_UPPER, verbose))
        fails++;
    if (test_ldl_solve(N, LB, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldl_solve(N, LB, ARMAS_UPPER, verbose))
        fails++;

    if (test_ldlpv_solve(N, 0, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldlpv_solve(N, 0, ARMAS_UPPER, verbose))
        fails++;
    if (test_ldlpv_solve(N, LB, ARMAS_LOWER, verbose))
        fails++;
    if (test_ldlpv_solve(N, LB, ARMAS_UPPER, verbose))
        fails++;

    exit(fails);
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
