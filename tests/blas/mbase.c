
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"

#define NAME "mbase"

DTYPE neg(DTYPE a)
{
    return -a;
}

int test_add(int M, int N, int flags, int verbose)
{
    armas_dense_t A0, B0, A1, B1, alpha;
    DTYPE __aval, n1, n0;
    armas_conf_t conf = *armas_conf_default();
    char *form =
        (flags & ARMAS_LOWER) ? "L" : ((flags & ARMAS_UPPER) ? "U" : "G");

    armas_init(&A0, M, N);
    armas_init(&B0, M, N);
    armas_init(&A1, M, N);
    armas_init(&B1, M, N);
    // singleton matrix
    armas_make(&alpha, 1, 1, 1, &__aval);

    armas_set_values(&A0, unitrand, flags);
    armas_set_values(&B0, unitrand, flags);
    // A1 = -A0; B1 = -B1
    armas_mcopy(&A1, &A0);
    armas_mcopy(&B1, &B0);
    armas_apply(&B1, neg, flags);
    armas_apply(&A1, neg, flags);

    // A0 = A0 + B0; A1 = |A1| + |B1|
    armas_add_elems(&A0, &B0, flags);
    armas_add_elems(&A1, &B1, flags | ARMAS_ABS);

    n1 = rel_error(&n0, &A0, &A1, ARMAS_NORM_INF, 0, &conf);
    int ok = isOK(n1, N) || n1 == 0.0;
    printf("%4s: A + B == |A| + |B| (form: %s)\n", PASS(ok), form);
    if (verbose > 0)
        printf("   || error || : %13e\n", n1);

    if (verbose > 1 && N < 10) {
        printf("error:\n");
        armas_printf(stdout, "%6.3f", &A0);
    }

    armas_release(&A0);
    armas_release(&B0);
    armas_release(&A1);
    armas_release(&B1);
    return 0;
}

int test_sub(int M, int N, int flags, int verbose)
{
    armas_dense_t A0, B0, A1, B1, alpha;
    DTYPE __aval, n1, n0;
    armas_conf_t conf = *armas_conf_default();
    char *form =
        (flags & ARMAS_LOWER) ? "L" : ((flags & ARMAS_UPPER) ? "U" : "G");

    armas_init(&A0, M, N);
    armas_init(&B0, M, N);
    armas_init(&A1, M, N);
    armas_init(&B1, M, N);
    // singleton matrix
    armas_make(&alpha, 1, 1, 1, &__aval);

    armas_set_values(&A0, unitrand, flags);
    armas_set_values(&B0, unitrand, flags);
    // A1 = -A0; B1 = -B1
    armas_mcopy(&A1, &A0);
    armas_mcopy(&B1, &B0);
    armas_apply(&B1, neg, flags);
    armas_apply(&A1, neg, flags);

    // A0 = A0 + B0; A1 = |A1| + |B1|
    armas_sub_elems(&A0, &B0, flags);
    armas_sub_elems(&A1, &B1, flags | ARMAS_ABS);

    n1 = rel_error(&n0, &A0, &A1, ARMAS_NORM_INF, 0, &conf);
    int ok = isOK(n1, N) || n1 == 0.0;
    printf("%4s: A - B == |A| - |B| (form: %s)\n", PASS(ok), form);
    if (verbose > 0)
        printf("   || error || : %13e\n", n1);

    if (verbose > 1 && N < 10) {
        printf("error:\n");
        armas_printf(stdout, "%6.3f", &A0);
    }

    armas_release(&A0);
    armas_release(&B0);
    armas_release(&A1);
    armas_release(&B1);
    return 0;
}

int test_gemm_abs(int M, int N, int flags, int verbose)
{
    armas_dense_t A0, B0, C0, A1, B1, C1, alpha;
    DTYPE __aval, n1, n0;
    armas_conf_t conf = *armas_conf_default();
    char *form =
        (flags & ARMAS_LOWER) ? "L" : ((flags & ARMAS_UPPER) ? "U" : "G");

    armas_init(&A0, M, N);
    armas_init(&B0, N, M);
    armas_init(&C0, M, M);
    armas_init(&A1, M, N);
    armas_init(&B1, N, M);
    armas_init(&C1, M, M);
    // singleton matrix
    armas_make(&alpha, 1, 1, 1, &__aval);

    armas_set_values(&A0, unitrand, flags);
    armas_set_values(&B0, unitrand, flags);
    // A1 = -A0; B1 = -B1
    armas_mcopy(&A1, &A0);
    armas_mcopy(&B1, &B0);
    armas_apply(&B1, neg, flags);
    armas_apply(&A1, neg, flags);
    if (verbose > 1 && N < 10) {
        printf("A0:\n");
        armas_printf(stdout, "%6.3f", &A0);
        printf("A1:\n");
        armas_printf(stdout, "%6.3f", &A1);
        printf("B0:\n");
        armas_printf(stdout, "%6.3f", &B0);
        printf("B1:\n");
        armas_printf(stdout, "%6.3f", &B1);
    }
    // A0 = A0 + B0; A1 = |A1| + |B1|
    armas_mult(0.0, &C0, 1.0, &A0, &B0, 0, &conf);
    armas_mult(0.0, &C1, 1.0, &A1, &B1, ARMAS_ABSA | ARMAS_ABSB, &conf);
    if (verbose > 1 && N < 10) {
        printf("C0:\n");
        armas_printf(stdout, "%6.3f", &C0);
        printf("C1:\n");
        armas_printf(stdout, "%6.3f", &C1);
    }

    n1 = rel_error(&n0, &C0, &C1, ARMAS_NORM_INF, 0, &conf);
    int ok = isOK(n1, N) || n1 == 0.0;
    printf("%4s: C + A* B == |A|*|B| (form: %s)\n", PASS(ok), form);
    if (verbose > 0)
        printf("   || error || : %13e\n", n1);

    if (verbose > 1 && N < 10) {
        //printf("error:\n"); armas_printf(stdout, "%6.3f", &C0);
    }

    armas_release(&A0);
    armas_release(&B0);
    armas_release(&A1);
    armas_release(&B1);
    armas_release(&C0);
    armas_release(&C1);
    return 0;
}

int main(int argc, char **argv)
{
    int opt;
    int M = 7;
    int N = 7;
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
        M = atoi(argv[optind]);
        N = atoi(argv[optind + 1]);
    } else if (optind < argc) {
        N = atoi(argv[optind]);
        M = N;
    }

    int fails = 0;
#if 0
    fails += test_add(M, N, 0, verbose);
    fails += test_add(N, N, ARMAS_UPPER, verbose);
    fails += test_add(N, N, ARMAS_LOWER, verbose);
    fails += test_sub(M, N, 0, verbose);
    fails += test_sub(N, N, ARMAS_UPPER, verbose);
    fails += test_sub(N, N, ARMAS_LOWER, verbose);
#endif
    fails += test_gemm_abs(M, N, 0, verbose);
    exit(fails);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
