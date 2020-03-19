
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "dtype.h"
#include "eft.h"
#include "testing.h"

#define FABS fabs
#define EPSILON DBL_EPSILON

static void seed()
{
    static int sinit = 0;
    if (!sinit) {
        srand48((long)time(0));
        sinit = 1;
    }
}

/*
 * Create test matrix for trsv such that A^-1*E = 1; A^-T*Et = 1
 *
 *   UPPER:                UPPER|TRANS:
 *   z  N  e  e -N   2z    N  0  0 -N -N    N
 *   .  w  N  e -N   2w    .  N  0  e  e    N
 *   .  .  N  0  0    N    .  .  N  N  e    N
 *   .  .  .  N  0    N    .  .  .  w  N   2w
 *   .  .  .  .  N    N    .  .  .  .  z   2z
 *
 * where z = (N-3)e = 2e; w = (N-4)*e = 1e
 *
 * Exact result is 1^T.
 */

void make_ext_trsv_data(
    int N, int right,
    armas_x_dense_t *Aa,
    armas_x_dense_t *Aat,
    armas_x_dense_t *Ee,
    armas_x_dense_t *Eet)
{
    armas_x_dense_t row, col, *A, *At, *E, *Et;
    DTYPE z;

    A  = right ? Aat : Aa;
    At = right ? Aa : Aat;
    E  = right ? Eet : Ee;
    Et = right ? Ee : Eet;

    for (int i = 0; i < N; i++) {
        armas_x_row(&row, A, i);
        armas_x_column(&col, At, N - 1 - i);
        if (i < N - 3) {
            for (int j = i; j < N; ++j) {
                armas_x_set_at(&row, j, EPS);
                armas_x_set_at(&col, N - 1 - j, EPS);
            }
            z = (N - 3 - i) * EPS;
            armas_x_set_at(&row, N - 1, -N);
            armas_x_set_at(&row, i, z);
            armas_x_set_at(&row, i + 1, N);
            armas_x_set_at(&col, 0, -N);
            armas_x_set_at(&col, N - 2 - i, N);
            armas_x_set_at(&col, N - 1 - i, z);

            armas_x_set_at(E, i, 2 * z);
            armas_x_set_at(Et, N - 1 - i, 2 * z);
        } else {
            armas_x_set_at(&row, i, N);
            armas_x_set_at(&col, N - 1 - i, N);
            armas_x_set_at(E, i, N);
            armas_x_set_at(Et, N - 1 - i, N);
        }
    }
}

void make_ext_trsm_matrix(
    int N, int flags,
    armas_x_dense_t *A, armas_x_dense_t *At,
    armas_x_dense_t *E, armas_x_dense_t *Et,
    armas_conf_t *cf)
{
    armas_x_dense_t e0, e1, c0, c1;

    if (flags & ARMAS_RIGHT) {
        armas_x_row(&e0, E, 0);
        armas_x_row(&e1, Et, 0);
    } else {
        armas_x_column(&e0, E, 0);
        armas_x_column(&e1, Et, 0);
    }
    make_ext_trsv_data(N, (flags & ARMAS_RIGHT) != 0, A, At, &e0, &e1);

    int K = (flags & ARMAS_RIGHT) != 0 ? E->rows : E->cols;
    for (int j = 1; j < K; ++j) {
        if (flags & ARMAS_RIGHT) {
            armas_x_row(&c0, E, j);
            armas_x_row(&c1, Et, j);
        } else {
            armas_x_column(&c0, E, j);
            armas_x_column(&c1, Et, j);
        }
        armas_x_copy(&c0, &e0, cf);
        armas_x_copy(&c1, &e1, cf);
    }
}

/*
 * Create test matrix for computing A*1 = E:
 *
 *   N e e e -N   3e   -N -N -N -N -N   -N
 *   . N e e -N   2e    .  N  e  e  e    0
 *   . . N e -N    e    .  .  N  e  e    e
 *   . . . N -N    0    .  .  .  N  e   2e
 *   . . . . -N   -N    .  .  .  .  N   3e
 *
 */

#define C 1000.0

void make_ext_trmv_data(
    int N,
    armas_x_dense_t *A,
    armas_x_dense_t *At,
    armas_x_dense_t *E,
    armas_x_dense_t *Et)
{
    armas_x_dense_t row, col;

    for (int i = 0; i < N; i++) {
        armas_x_row(&row, A, i);
        armas_x_column(&col, At, N - 1 - i);
        for (int j = i; j < N; ++j) {
            armas_x_set_at(&row, j, EPS);
            armas_x_set_at(&col, N - 1 - j, EPS);
        }
        armas_x_set_at(&row, i, N * C);
        armas_x_set_at(&row, N - 1, -N * C);
        armas_x_set_at(&col, N - 1 - i, N * C);
        armas_x_set_at(&col, 0, -N * C);

        if (i == 0) {
            armas_x_set_at(E, i, (N - 2 - i) * EPS);
            armas_x_set_at(Et, i, -N * C);
        } else if (i == N - 1) {
            armas_x_set_at(E, i, -N * C);
            armas_x_set_at(Et, i, (i - 1) * EPS);
        } else {
            armas_x_set_at(E, i, (N - 2 - i) * EPS);
            armas_x_set_at(Et, N - 1 - i, (N - 2 - i) * EPS);
        }
    }
}

void make_ext_trmm_matrix(
    int N, int flags,
    armas_x_dense_t *A, armas_x_dense_t *At,
    armas_x_dense_t *E, armas_x_dense_t *Et,
    armas_conf_t *cf)
{
    armas_x_dense_t e0, e1, c0, c1;

    if (flags & ARMAS_RIGHT) {
        armas_x_row(&e0, E, 0);
        armas_x_row(&e1, Et, 0);
    } else {
        armas_x_column(&e0, E, 0);
        armas_x_column(&e1, Et, 0);
    }
    make_ext_trmv_data(N, A, At, &e0, &e1);

    int K = (flags & ARMAS_RIGHT) != 0 ? E->rows : E->cols;
    for (int j = 1; j < K; ++j) {
        if (flags & ARMAS_RIGHT) {
            armas_x_row(&c0, E, j);
            armas_x_row(&c1, Et, j);
        } else {
            armas_x_column(&c0, E, j);
            armas_x_column(&c1, Et, j);
        }
        armas_x_copy(&c0, &e0, cf);
        armas_x_copy(&c1, &e1, cf);
    }
}

static DTYPE eps(int i, int j)
{
    return EPS;
}

static int rnd_index(int N)
{
    double e = round(drand48() * (N / 2.0));
    return (int)e;
}

void make_ext_matrix_data(
    armas_x_dense_t *A,
    DTYPE alpha,
    armas_x_dense_t *E,
    int flags)
{
    int k;
    armas_x_set_values(A, eps, 0);
    seed();
    if (flags & ARMAS_RIGHT) {
        for (int i = 0; i < A->cols; ++i) {
            k = rnd_index(A->rows);
            if (k == A->rows - 1 - k) {
                k += 1;
            }
            armas_x_set_unsafe(A, k, i, (DTYPE)A->rows);
            armas_x_set_unsafe(A, A->rows - 1 - k, i, (DTYPE)(-A->rows));
            armas_x_set_unsafe(E, i, 0, alpha * (A->rows - 2) * EPS);
        }
    } else {
        for (int i = 0; i < A->rows; ++i) {
            k = rnd_index(A->cols);
            if (k == A->cols - 1 - k) {
                k += 1;
            }
            armas_x_set_unsafe(A, i, k, (DTYPE)A->cols);
            armas_x_set_unsafe(A, i, A->cols - 1 - k, (DTYPE)(-A->cols));
            armas_x_set_unsafe(E, i, 0, alpha * (A->cols - 2) * EPS);
        }
    }
}
