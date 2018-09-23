
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
    armas_x_dense_t A0, A1, W;
    DTYPE n0, n1;
    char *blk = lb == 0 ? "unblk" : "  blk";
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&W, N, lb == 0 ? 1 : lb);
    armas_pivot_init(&P, N);

    armas_x_set_values(&A0, unitrand, 0);
    armas_x_mcopy(&A1, &A0);


    conf.lb = lb;
    armas_x_lufactor(&A1, &P, &conf);
    if (lb == 0 && N < 10) {
        printf("LU(A)\n");
        armas_x_printf(stdout, "%9.2e", &A1);
    }
    if (armas_x_luinverse(&A1, &P, &conf) < 0)
        printf("inverse.1 error: %d\n", conf.error);
    if (lb == 0 && N < 10) {
        printf("A.-1\n");
        armas_x_printf(stdout, "%9.2e", &A1);
    }

    armas_x_lufactor(&A1, &P, &conf);
    if (armas_x_luinverse(&A1, &P, &conf) < 0)
        printf("inverse.2 error: %d\n", conf.error);

    n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: %s.(A.-1).-1 == A\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&W);
    return fails;
}

int test_ident(int N, int lb, int verbose)
{
    armas_x_dense_t A0, A1, C, W, D;
    DTYPE n0;
    char *blk = lb == 0 ? "unblk" : "  blk";
    int ok, fails = 0;;
    armas_pivot_t P;
    armas_conf_t conf = *armas_conf_default();
    
    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&C, N, N);
    armas_x_init(&W, N, lb == 0 ? 1 : lb);
    armas_pivot_init(&P, N);

    armas_x_set_values(&A0, unitrand, 0);
    armas_x_mcopy(&A1, &A0);
    armas_x_diag(&D, &C, 0);

    conf.lb = lb;
    armas_x_lufactor(&A1, &P, &conf);
    armas_x_luinverse(&A1, &P, &conf);

    armas_x_mult(0.0, &C, 1.0, &A1, &A0, 0, &conf);
    armas_x_madd(&D, -1.0, 0);
    n0 = armas_x_mnorm(&C, ARMAS_NORM_INF, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: %s.A.-1*A == I\n", PASS(ok), blk);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&W);
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
