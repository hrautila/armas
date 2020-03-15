
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "testing.h"

#define SQRT2  1.4142135623730950
#define RSQRT2 0.7071067811865475


void make_butterfly(armas_x_dense_t * A, armas_x_dense_t * U, armas_conf_t *cf)
{
    //int valid = (A->rows & 0x1) == 0;
    int nb = A->rows / 2;
    armas_x_dense_t S, D;

    // set diagonal to U = [R; S]
    armas_x_diag(&D, A, 0);
    armas_x_mcopy(&D, U, 0, cf);
    // right side
    armas_x_subvector(&S, &D, nb, nb);
    // set top right to S
    armas_x_diag(&D, A, nb);
    armas_x_mcopy(&D, &S, 0, cf);
    // set bottom right to -S
    armas_x_mscale(&S, -ONE, 0, cf);
    // bottom left
    armas_x_diag(&D, A, -nb);
    armas_x_subvector(&S, U, 0, nb);
    armas_x_mcopy(&D, &S, 0, cf);
}

void clear_extra(armas_x_dense_t * U, int N, int unit)
{
    armas_x_dense_t T, D;
    // extra rows
    armas_x_submatrix(&T, U, N, 0, U->rows - N, U->cols);
    armas_x_set_values(&T, zero, 0);
    // clear extra columns
    armas_x_submatrix(&T, U, 0, N, N, U->cols - N);
    armas_x_set_values(&T, zero, 0);
    if (unit) {
        // make bottom right unit diagonal
        armas_x_submatrix(&T, U, N, N, U->rows - N, U->cols - N);
        if (armas_x_diag(&D, &T, 0))
            armas_x_set_values(&D, one, 0);
    }
}

void create_rbt_matrix(armas_x_dense_t * U2, armas_x_dense_t * U1,
                       armas_x_dense_t * S, int N, int Nd, int P, armas_conf_t *cf)
{
    armas_x_dense_t Uu, Su, S0, S1;

    armas_x_init(&S0, 0, 0);

    armas_x_gen_rbt(S);
    if (P == 2) {
        // full butterfly <N>
        armas_x_column(&S1, S, 1);
        make_butterfly(U2, &S1, cf);
        // top left butterfly <N/2>
        armas_x_column(&S0, S, 0);
        armas_x_submatrix(&Uu, U1, 0, 0, Nd / 2, Nd / 2);
        armas_x_subvector(&Su, &S0, 0, Nd / 2);
        make_butterfly(&Uu, &Su, cf);
        // bottom right butterfly <N/2>
        armas_x_submatrix(&Uu, U1, Nd / 2, Nd / 2, Nd / 2, Nd / 2);
        armas_x_subvector(&Su, &S0, Nd / 2, Nd / 2);
        make_butterfly(&Uu, &Su, cf);
        if (N != Nd) {
            clear_extra(U2, N, 0);
            clear_extra(U1, N, 0);
        }
    } else if (P == 1) {
        // full butterfly <N>
        armas_x_column(&S0, S, 0);
        make_butterfly(U2, &S0, cf);
        if (N != Nd) {
            clear_extra(U2, N, 0);
        }
    }
}

int test_update_rbt(int N, int P, int verbose)
{
    armas_x_dense_t U2, U1, S, Su, A0, A1, Ax, Au, Az;
    int Nd, mask, ok;
    DTYPE n0, n1;
    armas_conf_t conf = *armas_conf_default();

    if (P > 2)
        P = 2;

    Nd = N;
    mask = P == 1 ? 0x1 : 0x3;
    if ((N & mask) != 0) {
        Nd += 2 * P - (N & mask);
    }
    armas_x_init(&U2, Nd, Nd);
    armas_x_init(&U1, Nd, Nd);
    armas_x_init(&S, Nd, P);

    // U2 = <N>, U1 = <N/2>
    create_rbt_matrix(&U2, &U1, &S, N, Nd, P, &conf);

    armas_x_init(&A0, Nd, Nd);
    armas_x_set_values(&A0, unitrand, 0);
    if (N != Nd)
        clear_extra(&A0, N, 1);

    armas_x_init(&A1, Nd, Nd);
    armas_x_init(&Ax, Nd, Nd);
    armas_x_mcopy(&A1, &A0, 0, &conf);

    if (P == 2) {
        // RBT: <N/2>: U1.T*A1;  U1.T*A1*U1
        armas_x_mult(ZERO, &Ax, RSQRT2, &U1, &A1, ARMAS_TRANSA, &conf);
        armas_x_mult(ZERO, &A1, RSQRT2, &Ax, &U1, 0, &conf);

        clear_extra(&A1, N, 1);

        // RBT <N>:  U2.T*U1.T*A1*U1;  U2.T*U1.T*A1*U1*U2
        armas_x_mult(ZERO, &Ax, RSQRT2, &U2, &A1, ARMAS_TRANSA, &conf);
        armas_x_mult(ZERO, &A1, RSQRT2, &Ax, &U2, 0, &conf);
    } else if (P == 1) {
        // RBT <N>: U2.T*A1; U2.T*A*U2
        armas_x_mult(ZERO, &Ax, RSQRT2, &U2, &A1, ARMAS_TRANSA, &conf);
        armas_x_mult(ZERO, &A1, RSQRT2, &Ax, &U2, 0, &conf);
    }
    // compute RBT 
    if (N != Nd) {
        armas_x_submatrix(&Au, &A0, 0, 0, N, N);
        armas_x_submatrix(&Su, &S, 0, 0, N, P);
        armas_x_update2_rbt(&Au, &Su, &Su, &conf);
    } else {
        armas_x_update2_rbt(&A0, &S, &S, &conf);
    }

    if (verbose > 1 && Nd < 10) {
        armas_x_submatrix(&Au, &A0, 0, 0, N, N);
        printf("update_rbt(A0, U)\n");
        armas_x_printf(stdout, "%6.3f", &Au);
        armas_x_submatrix(&Au, &A1, 0, 0, N, N);
        printf("S.T*A0*S\n");
        armas_x_printf(stdout, "%6.3f", &Au);
    }

    armas_x_submatrix(&Au, &A1, 0, 0, N, N);
    armas_x_submatrix(&Az, &A0, 0, 0, N, N);
    n0 = rel_error(&n1, &Au, &Az, ARMAS_NORM_INF, 0, &conf);
    ok = n0 == ZERO || isOK(n0, N);
    printf("%4s: rbt(A, U, U) == U.T*A*U [N=%d, Nd=%d]\n", PASS(ok), N, Nd);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_x_release(&U1);
    armas_x_release(&U2);
    armas_x_release(&S);
    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&Ax);

    return 1 - ok;
}

#if FLOAT32
#define MAX_ERROR  1e-6
#else
#define MAX_ERROR  1e-12
#endif

int test_rbt_lufactor(int N, int P, int verbose)
{
    armas_x_dense_t A0, A1, B0, U, V, B1, X1;
    DTYPE n0, n1;
    armas_pivot_t p;
    int ok, fails = 0;
    armas_conf_t conf = *armas_conf_default();

    armas_x_init(&A0, N, N);
    armas_x_init(&A1, N, N);
    armas_x_init(&U, N, P);
    armas_x_init(&V, N, P);
    armas_pivot_init(&p, N);

    armas_x_init(&B0, N, N / 2);
    armas_x_init(&B1, N, N / 2);
    armas_x_init(&X1, N, N / 2);

    armas_x_set_values(&A0, unitrand, 0);
    // make it nearly singular ..
    // ( 2*y  1     ... )      ( 2*y  1 ... )
    // (  y  y/2+e  ... ) -->  ( y/2  e ... )
    // (  ....          )      (  .....     )
    armas_x_set(&A0, 0, 0, 2.0);
    armas_x_set(&A0, 0, 1, ONE);
    armas_x_set(&A0, 1, 0, ONE);
    armas_x_set(&A0, 1, 1, 0.5 + EPS);
    armas_x_mcopy(&A1, &A0, 0, &conf);
    armas_x_set_values(&B0, unitrand, 0);
    armas_x_mcopy(&B1, &B0, 0, &conf);
    if (verbose > 2 && N < 10) {
        printf("A:\n");
        armas_x_printf(stdout, "%6.3f", &A0);
    }

    armas_x_gen_rbt(&U);
    armas_x_gen_rbt(&V);
    // U.T*A*V
    armas_x_update2_rbt(&A1, &U, &V, &conf);
    armas_x_lufactor(&A1, (armas_pivot_t *) 0, &conf);
    // U.T*B
    armas_x_mcopy(&X1, &B1, 0, &conf);
    armas_x_mult_rbt(&X1, &U, ARMAS_LEFT | ARMAS_TRANS, &conf);
    // solve
    armas_x_lusolve(&X1, &A1, (armas_pivot_t *) 0, 0, &conf);
    // X = V*B
    armas_x_mult_rbt(&X1, &V, ARMAS_LEFT, &conf);
    armas_x_mult(ZERO, &B1, ONE, &A0, &X1, 0, &conf);
    n0 = rel_error(&n1, &B1, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = n0 == ZERO || isFINE(n0, N * MAX_ERROR);
    fails += 1 - ok;
    printf("%4s: || X - V*genp(U.T*A*V, U*B) ||\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error/rbt  || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 2 && N < 10) {
        printf("genp(U.T*A*V):\n");
        armas_x_printf(stdout, "%13e", &A1);
    }
    // same without randomization....
    armas_x_mcopy(&A1, &A0, 0, &conf);
    armas_x_lufactor(&A1, (armas_pivot_t *) 0, &conf);
    armas_x_mcopy(&X1, &B0, 0, &conf);
    armas_x_lusolve(&X1, &A1, (armas_pivot_t *) 0, 0, &conf);

    armas_x_mult(ZERO, &B1, ONE, &A0, &X1, 0, &conf);
    n0 = rel_error(&n1, &B1, &B0, ARMAS_NORM_INF, 0, &conf);
    if (verbose > 0) {
        printf("   || rel error/genp || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 2 && N < 10) {
        printf("genp(A):\n");
        armas_x_printf(stdout, "%13e", &A1);
    }
    // same with pivoting....
    armas_x_mcopy(&A1, &A0, 0, &conf);
    armas_x_lufactor(&A1, &p, &conf);
    armas_x_mcopy(&X1, &B0, 0, &conf);
    armas_x_lusolve(&X1, &A1, &p, 0, &conf);

    armas_x_mult(ZERO, &B1, ONE, &A0, &X1, 0, &conf);
    n0 = rel_error(&n1, &B1, &B0, ARMAS_NORM_INF, 0, &conf);
    if (verbose > 0) {
        printf("   || rel error/gepp || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 2 && N < 10) {
        printf("gepp(A):\n");
        armas_x_printf(stdout, "%13e", &A1);
    }

    armas_x_release(&A0);
    armas_x_release(&A1);
    armas_x_release(&B0);
    armas_x_release(&B1);
    armas_x_release(&X1);
    armas_x_release(&U);
    armas_x_release(&V);

    return fails;
}


int main(int argc, char **argv)
{
    int N, P, opt, verbose = 1;

    N = 144;
    P = 2;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            break;
        }
    }

    switch (argc - optind) {
    case 2:
        N = atoi(argv[optind]);
        P = atoi(argv[optind + 1]);
        break;
    case 1:
        N = atoi(argv[optind]);
        break;
    default:
        break;
    }
    int fails = 0;

    if (test_update_rbt(N, P, verbose))
        fails++;
    if (test_update_rbt(N - 1, P, verbose))
        fails++;
    if (test_rbt_lufactor(N, P, verbose))
        fails++;
    if (test_rbt_lufactor(N - 1, P, verbose))
        fails++;
    return fails;
}


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
