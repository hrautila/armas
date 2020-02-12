
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t armas_x_dense_t;
typedef float DTYPE;

#define armas_x_init       armas_s_init
#define armas_x_set_values armas_s_set_values
#define armas_x_mult       armas_s_mult
#define armas_x_transpose  armas_s_transpose
#define armas_x_release    armas_s_release
#define armas_x_mult_trm   armas_s_mult_trm
#define armas_x_mcopy      armas_s_mcopy

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t armas_x_dense_t;
typedef double DTYPE;

#define armas_x_init       armas_d_init
#define armas_x_set_values armas_d_set_values
#define armas_x_mult       armas_d_mult
#define armas_x_transpose  armas_d_transpose
#define armas_x_release    armas_d_release
#define armas_x_mult_trm   armas_d_mult_trm
#define armas_x_mcopy      armas_d_mcopy

#endif
#include "helper.h"

int main(int argc, char **argv)
{

    armas_conf_t conf;
    armas_x_dense_t C, B0, A, B;

    int ok, opt, fails = 0;
    int N = 600;
    int verbose = 1;
    DTYPE alpha = 1.0, n0, n1;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        default:
            fprintf(stderr, "usage: test_trmm [-v] [size]\n");
            exit(1);
        }
    }

    if (optind < argc)
        N = atoi(argv[optind]);

    conf = *armas_conf_default();

    armas_x_init(&C, N, N);
    armas_x_init(&A, N, N);
    armas_x_init(&B, N, N);
    armas_x_init(&B0, N, N);

    armas_x_set_values(&C, zero, ARMAS_NULL);
    armas_x_set_values(&A, zero, ARMAS_NULL);
    armas_x_set_values(&A, unitrand, ARMAS_UPPER);
    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_mcopy(&B0, &B);

    // B = A*B
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_LEFT, &conf);
    armas_x_mult(0.0, &C, alpha, &A, &B0, ARMAS_NULL, &conf);

    n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmm(B, A, L|U)   == gemm(TriU(A), B)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_set_values(&B0, one, ARMAS_NULL);
    armas_x_set_values(&C, zero, ARMAS_NULL);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_LEFT | ARMAS_TRANSA,
                     &conf);
    armas_x_mult(0.0, &C, alpha, &A, &B0, ARMAS_TRANSA, &conf);

    n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmm(B, A, L|U|T) == gemm(TriU(A).T, B)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_set_values(&B0, one, ARMAS_NULL);
    armas_x_set_values(&C, zero, ARMAS_NULL);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_RIGHT, &conf);
    armas_x_mult(0.0, &C, alpha, &B0, &A, ARMAS_NULL, &conf);

    n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmm(B, A, R|U)   == gemm(B, TriU(A))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_set_values(&B, one, ARMAS_NULL);
    armas_x_set_values(&B0, one, ARMAS_NULL);
    armas_x_set_values(&C, zero, ARMAS_NULL);
    armas_x_mult_trm(&B, alpha, &A, ARMAS_UPPER | ARMAS_RIGHT | ARMAS_TRANSA,
                     &conf);
    armas_x_mult(0.0, &C, alpha, &B0, &A, ARMAS_TRANSB, &conf);

    n0 = rel_error(&n1, &B, &C, ARMAS_NORM_ONE, ARMAS_NONE, &conf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmm(B, A, R|U|T) == gemm(B, TriU(A).T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
