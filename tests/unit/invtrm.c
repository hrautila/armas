
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "test"

#if FLOAT32
#define __ERROR 1e-4
#else
#define __ERROR 1e-8
#endif

__Dtype unit100(int i, int j)
{
    return 100.0*unitrand(i, j);
}

int test_equal(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, A2;
    __Dtype n0, n1;
    char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
    int ok, fails = 0;;
    
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&A2, N, N);

    matrix_set_values(&A0, unit100, flags);
    matrix_mcopy(&A1, &A0);
    matrix_mcopy(&A2, &A0);

    // unblocked; inverse twice
    conf.lb = 0;
    matrix_inverse_trm(&A1, flags, &conf);
    matrix_inverse_trm(&A1, flags, &conf);
    n0 = rel_error(&n1, &A1, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: [%c] unblk.(A.-1).-1 == A\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    // unblocked; inverse twice
    conf.lb = lb;
    matrix_inverse_trm(&A2, flags, &conf);
    matrix_inverse_trm(&A2, flags, &conf);
    n0 = rel_error(&n1, &A2, &A0, ARMAS_NORM_INF, 0, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: [%c]   blk.(A.-1).-1 == A\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }
    
    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&A2);
    return fails;
}

int test_ident(int N, int lb, int flags, int verbose)
{
    __Matrix A0, A1, A2, C, C0, D;
    __Dtype n0, n1;
    char uplo = flags & ARMAS_UPPER ? 'U' : 'L';
    int ok, fails = 0;;
    
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&A2, N, N);
    matrix_init(&C, N, N);

    // make unit matrix C0
    matrix_init(&C0, N, N);
    matrix_diag(&D, &C0, 0);
    matrix_madd(&D, 1.0, 0);
    
    matrix_set_values(&A0, unit100, flags);
    matrix_mcopy(&A1, &A0);
    matrix_mcopy(&A2, &A0);

    // unblocked
    conf.lb = 0;
    matrix_inverse_trm(&A1, flags, &conf);

    // C = A*A.-1
    matrix_mult(&C, &A0, &A1, 1.0, 0.0, 0, &conf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_INF, ARMAS_NONE, &conf);
    ok = isFINE(n0, N*__ERROR);
    //ok = isOK(n0, N);
    fails += 1 - ok;
    printf("%s: [%c] unblk.A.-1*A == I\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    // blocked
    conf.lb = lb;
    matrix_inverse_trm(&A2, flags, &conf);

    matrix_mult(&C, &A0, &A2, 1.0, 0.0, 0, &conf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_INF, ARMAS_NONE, &conf);
    ok = isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: [%c]  blk.A.-1*A == I\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }
    n0 = rel_error(&n1, &A1, &A2, ARMAS_NORM_INF, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isFINE(n0, N*__ERROR);
    fails += 1 - ok;
    printf("%s: [%c] unblk.A.-1 == blk.A.-1\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("  || rel error ||: %e [%d]\n", n0, ndigits(n0));
    }

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&A2);
    matrix_release(&C);
    matrix_release(&C0);
    return fails;
}    


int main(int argc, char **argv)
{
    int opt;
    int N = 32;
    int LB = 8;
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
        LB = N/10 < 4 ? 4 : (N/10) & ~0x1;
    }

    int fails = 0;
    if (test_ident(N, LB, ARMAS_UPPER, verbose) != 0)
        fails++;
    if (test_ident(N, LB, ARMAS_LOWER, verbose) != 0)
        fails++;
    if (test_equal(N, LB, ARMAS_UPPER, verbose) != 0)
        fails++;
    if (test_equal(N, LB, ARMAS_LOWER, verbose) != 0)
        fails++;

    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
