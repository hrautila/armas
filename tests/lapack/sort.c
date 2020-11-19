
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "testing.h"
/*
#if FLOAT32
#define _ERROR 1e-6
#else
#define _ERROR 1e-14
#endif
*/
#define NAME "evd"


static
int is_asc_sorted(armas_dense_t *x)
{
    DTYPE v0, v1;
    int result = 1;
    for (int i = 1; result && i < armas_size(x); i++) {
        v0 = armas_get_at_unsafe(x, i-1);
        v1 = armas_get_at_unsafe(x, i);
        result &= v0 <= v1;
    }
    return result;
}

static
int is_desc_sorted(armas_dense_t *x)
{
    DTYPE v0, v1;
    int result = 1;
    for (int i = 1; result && i < armas_size(x); i++) {
        v0 = armas_get_at_unsafe(x, i-1);
        v1 = armas_get_at_unsafe(x, i);
        result &= v0 >= v1;
    }
    return result;
}

static
int match(armas_dense_t *x, armas_dense_t *y)
{
    DTYPE v0, v1;
    int result = 1;
    for (int i = 1; result && i < armas_size(x); i++) {
        v0 = armas_get_at_unsafe(y, i);
        v1 = armas_get_at_unsafe(x, i);
        result &= v0 == v1;
    }
    return result;
}

int test_1(int N, int verbose)
{
    armas_dense_t A, c, row, row1, A0;
    int err, ok;

    armas_init(&A, N, N);
    for (int j = 0; j < N; j++) {
        armas_column(&c, &A, j);
        for (int i = 0; i < N; i++)
            armas_set_at_unsafe(&c, i, (DTYPE)(j + 1));
    }
    err = 0;
    armas_row(&row, &A, 0);
    armas_row(&row1, &A, 1);
    armas_submatrix(&A0, &A, 1, 0, N-1, N);

    armas_sort_eigenvec(&row, &A0, __nil, __nil, ARMAS_DESC);
    ok = is_desc_sorted(&row);
    ok &= match(&row, &row1);
    printf(" descending sort: %s\n", PASS(ok));
    if (N < 15)
        armas_printf(stdout, "%3.1f", &row);
    err += 1 - ok;

    armas_sort_eigenvec(&row, &A0, __nil, __nil, ARMAS_ASC);
    ok = is_asc_sorted(&row);
    ok &= match(&row, &row1);
    printf("  ascending sort: %s\n", PASS(ok));
    if (N < 15)
        armas_printf(stdout, "%3.1f", &row);
    err += 1 - ok;

    armas_qsort_vec(&row, ARMAS_DESC);
    ok = is_desc_sorted(&row);
    printf("descending qsort: %s\n", PASS(ok));
    if (N < 15)
        armas_printf(stdout, "%3.1f", &row);
    err += 1 - ok;

    armas_qsort_vec(&row, ARMAS_ASC);
    ok = is_asc_sorted(&row);
    printf(" ascending qsort: %s\n", PASS(ok));
    if (N < 15)
        armas_printf(stdout, "%3.1f", &row);
    err += 1 - ok;

    armas_release(&A);
    return ok;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 313;
    int verbose = 1;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: %s [-v] [N]\n", NAME);
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;

    if (!test_1(N, verbose))
        fails++;

    exit(fails);
}
