
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "inverse"

#if FLOAT32
#define __ERROR 1e-5
#else
#define __ERROR 1e-12
#endif


int test_equal(int N, int lb, int verbose)
{
    __Matrix A0, A1, W;
    __Dtype n0, n1;
    char *blk = lb == 0 ? "unblk" : "  blk";
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&W, N, lb == 0 ? 1 : lb);
    armas_pivot_init(&P, N);

    matrix_set_values(&A0, unitrand, 0);
    matrix_mcopy(&A1, &A0);


    conf.lb = lb;
    matrix_lufactor(&A1, &P, &conf);
    matrix_inverse(&A1, &W, &P, &conf);

    matrix_lufactor(&A1, &P, &conf);
    matrix_inverse(&A1, &W, &P, &conf);

    n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: %s.(A.-1).-1 == A\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&W);
    return fails;
}

int test_ident(int N, int lb, int verbose)
{
    __Matrix A0, A1, C, W, D;
    __Dtype n0;
    char *blk = lb == 0 ? "unblk" : "  blk";
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&C, N, N);
    matrix_init(&W, N, lb == 0 ? 1 : lb);
    armas_pivot_init(&P, N);

    matrix_set_values(&A0, unitrand, 0);
    matrix_mcopy(&A1, &A0);
    matrix_diag(&D, &C, 0);

    conf.lb = lb;
    matrix_lufactor(&A1, &P, &conf);
    matrix_inverse(&A1, &W, &P, &conf);

    matrix_mult(&C, &A1, &A0, 1.0, 0.0, 0, &conf);
    matrix_madd(&D, -1.0, 0);
    n0 = matrix_mnorm(&C, ARMAS_NORM_INF, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: %s.A.-1*A == I\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&W);
    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 515;
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
        N = atoi(argv[optind+1]);
        LB = atoi(argv[optind+1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        LB = N > 32 ? 16 : 8;
    }

    int fails = 0;
    if (test_equal(N, 0, verbose) != 0) 
        fails++;
    if (test_equal(N, LB, verbose) != 0) 
        fails++;
    if (test_ident(N, 0, verbose) != 0) 
        fails++;
    if (test_ident(N, LB, verbose) != 0) 
        fails++;
    
    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
