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
    armas_x_dense_t A0, A1, C, D, D0, W;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1;
    int ok, flags1, flags2;
    char *fact = flags & ARMAS_LOWER ? "L*D*L.T" : "U*D*U.T";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&W, lb == 0 ? N : lb*N, 1);
    armas_x_diag(&D, &C, 0);
    armas_x_madd(&D, 1.0, 0);

    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0);

    conf.lb = lb;
    armas_x_ldlfactor(&A0, &W, ARMAS_NOPIVOT, flags, &conf);
    armas_x_diag(&D0, &A0, 0);

    flags1 = flags|ARMAS_UNIT|ARMAS_RIGHT;
    flags2 = flags|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT;

    // C = I*A; C = C*D; C = C*L^T
    armas_x_mult_trm(&C, 1.0, &A0, flags1, &conf);
    armas_x_mult_diag(&C, &D0, 1.0, flags|ARMAS_RIGHT, &conf);
    armas_x_mult_trm(&C, 1.0, &A0, flags2, &conf);
    armas_x_make_trm(&C, flags);

    n0 = rel_error(&n1, &C, &A1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.%s = A\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&C);
    armas_x_release(&W);
    return 1 - ok;
}

int test_ldl_solve(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, B, B0, W;
    armas_conf_t conf = *armas_conf_default();
    DTYPE n0, n1;
    int ok;
    char *fact = flags & ARMAS_LOWER ? "L*D*L.T" : "U*D*U.T";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&B0, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&W, lb == 0 ? N : lb*N, 1);

    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0);

    armas_x_set_values(&B0, unitrand, 0);
    // B = A*B0
    armas_x_mult_sym(0.0, &B, 1.0, &A0, &B0, flags|ARMAS_LEFT, &conf);

    conf.lb = lb;
    armas_x_ldlfactor(&A0, &W, ARMAS_NOPIVOT, flags, &conf);
    armas_x_ldlsolve(&B, &A0, ARMAS_NOPIVOT, flags, &conf);

    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.(%s).-1*B = X\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&B);
    armas_x_release(&B0);
    armas_x_release(&W);
    return 1 - ok;
}

// with pivoting; verify A = L*D*L.T or A = U*D*U.T
int test_ldlpv(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, C, D, D0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P;
    DTYPE n0, n1;
    int ok, flags1, flags2;
    char *fact = flags & ARMAS_LOWER ? "P^T*(LDL^T)*P" : "P^T*(UDU^T)*P";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&W, lb == 0 ? N : lb*N, 1);
    armas_x_diag(&D, &C, 0);
    armas_x_madd(&D, 1.0, 0);
    armas_pivot_init(&P, N);
    
    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0);
    if (N < 10) {
        printf("A:\n"); armas_x_printf(stdout, "%6.3f", &A0);
    }

    conf.lb = lb;
    armas_x_ldlfactor(&A0, &W, &P, flags, &conf);
    armas_x_diag(&D0, &A0, 0);

    flags1 = flags|ARMAS_UNIT|ARMAS_RIGHT;
    flags2 = flags|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT;

    // C = ((I*L)*D)*L^T
    armas_x_mult_trm(&C, 1.0, &A0, flags1, &conf);
    armas_x_mult_diag(&C, &D0, 1.0, flags|ARMAS_RIGHT, &conf);
    armas_x_mult_trm(&C, 1.0, &A0, flags2, &conf);
    armas_x_make_trm(&C, flags);
    if (flags & ARMAS_LOWER) {
        armas_x_pivot(&C, &P, ARMAS_PIVOT_LOWER|ARMAS_PIVOT_BACKWARD, &conf);
    } else {
        armas_x_pivot(&C, &P, ARMAS_PIVOT_UPPER|ARMAS_PIVOT_FORWARD, &conf);
    }

    if (N < 10) {
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P);
        printf("%s:\n", fact); armas_x_printf(stdout, "%6.3f", &C);
    }
    n0 = rel_error(&n1, &C, &A1, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.%s = A\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&C);
    armas_x_release(&W);
    armas_pivot_release(&P);
    return 1 - ok;
}

// with pivoting; verify (L*D*L.T).T = U*D*U.T
int test_ldlpv_transpose(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, A2, C0, C1, D1, D0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P0, P1;
    DTYPE n0, n1;
    int ok;
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&A2, N, N);
    armas_x_init(&C0, N, N);
    armas_x_init(&C1, N, N);
    armas_x_init(&W, lb == 0 ? N : lb*N, 1);
    armas_x_diag(&D0, &C0, 0);
    armas_x_madd(&D0, 1.0, 0);
    armas_x_mcopy(&C1, &C0);
    armas_x_diag(&D1, &C1, 0);
    armas_pivot_init(&P0, N);
    armas_pivot_init(&P1, N);
    
    armas_x_set_values(&A0, unitrand, ARMAS_SYMM);
    armas_x_mcopy(&A1, &A0);
    armas_x_mcopy(&A2, &A0);
    armas_x_make_trm(&A0, ARMAS_LOWER);
    armas_x_make_trm(&A1, ARMAS_UPPER);
    if (N < 10) {
        printf("A:\n"); armas_x_printf(stdout, "%6.3f", &A2);
    }

    conf.lb = lb;
    armas_x_ldlfactor(&A0, &W, &P0, ARMAS_LOWER, &conf);
    armas_x_diag(&D0, &A0, 0);
    armas_x_ldlfactor(&A1, &W, &P1, ARMAS_UPPER, &conf);
    armas_x_diag(&D1, &A1, 0);

    // C = ((I*L)*D)*L.T
    armas_x_mult_trm(&C0, 1.0, &A0, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    armas_x_mult_diag(&C0, &D0, 1.0, ARMAS_LOWER|ARMAS_RIGHT, &conf);
    armas_x_mult_trm(&C0, 1.0, &A0, ARMAS_LOWER|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    armas_x_make_trm(&C0, ARMAS_LOWER);
    armas_x_pivot(&C0, &P0, ARMAS_PIVOT_LOWER|ARMAS_PIVOT_BACKWARD, &conf);

    if (N < 10) {
        printf("(1) P^T*(LDL^T)*P:\n"); armas_x_printf(stdout, "%6.3f", &C0);
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P0);
    }
    // C = ((I*U)*D)*U.T
    armas_x_mult_trm(&C1, 1.0, &A1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    armas_x_mult_diag(&C1, &D1, 1.0, ARMAS_UPPER|ARMAS_RIGHT, &conf);
    armas_x_mult_trm(&C1, 1.0, &A1, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT|ARMAS_RIGHT, &conf);
    armas_x_make_trm(&C1, ARMAS_UPPER);
    armas_x_pivot(&C1, &P1, ARMAS_PIVOT_UPPER|ARMAS_PIVOT_FORWARD, &conf);

    if (N < 10) {
        printf("(2) P^T*(UDU^T)*P:\n"); armas_x_printf(stdout, "%6.3f", &C1);
        printf("P:\n"); armas_pivot_printf(stdout, "%d", &P1);
    }

    n0 = rel_error(&n1, &C0, &C1, ARMAS_NORM_INF, ARMAS_TRANSB, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.P^T*(LDL^T)*P = transpose(%s.P^T*(UDU^T)*P)\n", PASS(ok), blk, blk);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&A2);
    armas_x_release(&C0);
    armas_x_release(&C1);
    armas_x_release(&W);
    armas_pivot_release(&P0);
    armas_pivot_release(&P1);
    return 1 - ok;
}

int test_ldlpv_solve(int N, int lb, int flags, int verbose)
{
    armas_x_dense_t A0, A1, B, B0, W;
    armas_conf_t conf = *armas_conf_default();
    armas_pivot_t P0;
    DTYPE n0, n1;
    int ok;
    char *fact = flags & ARMAS_LOWER ? "P^T*(LDL^T)*P" : "P^T*(UDU^T)*P";
    char *blk = lb == 0 ? "unblk" : "  blk";
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&B0, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&W, lb == 0 ? N : lb*N, 1);
    armas_pivot_init(&P0, N);
    
    armas_x_set_values(&A0, unitrand, flags);
    armas_x_mcopy(&A1, &A0);

    armas_x_set_values(&B0, unitrand, 0);
    // B = A*B0
    armas_x_mult_sym(0.0, &B, 1.0, &A0, &B0, flags|ARMAS_LEFT, &conf);

    if (N < 10) {
        printf("B0:\n"); armas_x_printf(stdout, "%6.3f", &B0);
    }

    conf.lb = lb;
    armas_x_ldlfactor(&A0, &W, &P0, flags, &conf);
    armas_x_ldlsolve(&B, &A0, &P0, flags, &conf);

    if (N < 10) {
        printf("(%s)^-1*B:\n", fact); armas_x_printf(stdout, "%6.3f", &B);
    }
    n0 = rel_error(&n1, &B, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    printf("%s : %s.(%s)^-1*B = X\n", PASS(ok), blk, fact);
    printf("   || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&B);
    armas_x_release(&B0);
    armas_x_release(&W);
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
