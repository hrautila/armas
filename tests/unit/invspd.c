
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#if FLOAT32
#define __ERROR 1e-4
#else
#define __ERROR 1e-8
#endif

#define NAME "invspd"

int test_ident(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, A2, C, W, D;
    __Dtype n0;
    char *blk = lb == 0 ? "unblk" : "  blk";
    char uplo = flags & ARMAS_LOWER ? 'L' : 'U';
    int ok;
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&A2, N, N);
    matrix_init(&C, N, N);
    matrix_init(&W, N, lb == 0 ? 1 : lb);

    // symmetric positive definite matrix A*A.T
    matrix_set_values(&A0, unitrand, 0);
    matrix_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
    matrix_make_trm(&A1, flags);
    matrix_mcopy(&A0, &A1);

    // identity; D = diag(C)
    matrix_diag(&D, &C, 0);
    matrix_set_values(&D, one, 0);

    conf.lb = lb;
    matrix_cholfactor(&A1, flags, &conf);
    matrix_inverse_spd(&A1, &W, flags, &conf);

    // A2 = A1*I
    matrix_mult_sym(&A2, &A1, &C, 1.0, 0.0, flags|ARMAS_LEFT, &conf);
    matrix_mult_sym(&C, &A0, &A2, 1.0, 0.0, flags, &conf);

    // diag(C) -= 1.0
    matrix_madd(&D, -1.0, 0);
    n0 = matrix_mnorm(&C, ARMAS_NORM_INF, &conf);
    ok = isFINE(n0, N*__ERROR);

    printf("%s: %c %s.A.-1*A == I\n", PASS(ok), uplo, blk);
    printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    
    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&A2);
    matrix_release(&C);
    matrix_release(&W);
    return 1 - ok;
}

int test_equal(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, W;
    __Dtype n0;
    char *blk = lb == 0 ? "unblk" : "  blk";
    char uplo = flags & ARMAS_LOWER ? 'L' : 'U';
    int ok;
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&W, N, lb == 0 ? 1 : lb);

    // symmetric positive definite matrix A*A.T
    matrix_set_values(&A0, unitrand, 0);
    matrix_mult(&A1, &A0, &A0, 1.0, 0.0, ARMAS_TRANSB, &conf);
    matrix_make_trm(&A1, flags);
    matrix_mcopy(&A0, &A1);

    conf.lb = lb;
    matrix_cholfactor(&A1, flags, &conf);
    matrix_inverse_spd(&A1, &W, flags, &conf);

    matrix_cholfactor(&A1, flags, &conf);
    matrix_inverse_spd(&A1, &W, flags, &conf);
    
    n0 = rel_error(&n0, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    
    printf("%s: %c %s.(A.-1).-1 == A\n", PASS(ok), uplo, blk);
    printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&W);
    return 1 - ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 177;
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
        N = atoi(argv[optind]);
        LB = atoi(argv[optind+1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;
    if (test_ident(N, 0, ARMAS_LOWER, verbose))
        fails ++;
    if (test_ident(N, LB, ARMAS_LOWER, verbose))
        fails ++;
    if (test_ident(N, 0, ARMAS_UPPER, verbose))
        fails ++;
    if (test_ident(N, LB, ARMAS_UPPER, verbose))
        fails ++;
    if (test_equal(N, 0, ARMAS_LOWER, verbose))
        fails ++;
    if (test_equal(N, LB, ARMAS_LOWER, verbose))
        fails ++;
    if (test_equal(N, 0, ARMAS_UPPER, verbose))
        fails ++;
    if (test_equal(N, LB, ARMAS_UPPER, verbose))
        fails ++;
    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
