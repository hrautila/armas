
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "../unit/testing.h"

#define __SQRT2  1.4142135623730950
#define __RSQRT2 0.7071067811865475


void make_butterfly(__Matrix *A, __Matrix *U)
{
    //int valid = (A->rows & 0x1) == 0;
    int nb = A->rows / 2;
    __Matrix S, D;
    
    // set diagonal to U = [R; S]
    matrix_diag(&D, A, 0);
    matrix_mcopy(&D, U);
    // right side
    matrix_subvector(&S, &D, nb, nb);
    // set top right to S
    matrix_diag(&D, A, nb);
    matrix_mcopy(&D, &S);
    // set bottom right to -S
    matrix_mscale(&S, -1.0, 0);
    // bottom left
    matrix_diag(&D, A, -nb);
    matrix_subvector(&S, U, 0, nb);
    matrix_mcopy(&D, &S);
}

void clear_extra(__Matrix *U, int N, int unit)
{
    __Matrix T, D;
    // extra rows
    matrix_submatrix(&T, U, N, 0, U->rows-N, U->cols);
    matrix_set_values(&T, zero, 0);
    // clear extra columns
    matrix_submatrix(&T, U, 0, N, N, U->cols-N);
    matrix_set_values(&T, zero, 0);
    if (unit) {
        // make bottom right unit diagonal
        matrix_submatrix(&T, U, N, N, U->rows-N, U->cols-N);
        if (matrix_diag(&D, &T, 0))
            matrix_set_values(&D, one, 0);
    }
}

void create_rbt_matrix(__Matrix *U2, __Matrix *U1, __Matrix *S, int N, int Nd, int P)
{
    __Matrix Uu, Su, S0, S1;
    
    matrix_init(&S0, 0, 0);

    matrix_gen_rbt(S);
    if (P == 2) {
        // full butterfly <N>
        matrix_column(&S1, S, 1);
        make_butterfly(U2, &S1);
        // top left butterfly <N/2>
        matrix_column(&S0, S, 0);
        matrix_submatrix(&Uu, U1,  0, 0, Nd/2, Nd/2);
        matrix_subvector(&Su, &S0, 0, Nd/2);
        make_butterfly(&Uu, &Su);
        // bottom right butterfly <N/2>
        matrix_submatrix(&Uu, U1,  Nd/2, Nd/2, Nd/2, Nd/2);
        matrix_subvector(&Su, &S0, Nd/2, Nd/2);
        make_butterfly(&Uu, &Su);
        if (N != Nd) {
            clear_extra(U2, N, 0);
            clear_extra(U1, N, 0);
        }
    } else if (P == 1) {
        // full butterfly <N>
        matrix_column(&S0, S, 0);
        make_butterfly(U2, &S0);
        if (N != Nd) {
            clear_extra(U2, N, 0);
        }
    }
}

int test_update_rbt(int N, int P, int verbose)
{
    __Matrix U2, U1, S, Su, A0, A1, Ax, Au, Az;
    int Nd, mask, ok;
    __Dtype n0, n1;
    armas_conf_t conf = *armas_conf_default();
    
    if (P > 2)
        P = 2;

    Nd = N;
    mask = P == 1 ? 0x1 : 0x3;
    if ((N & mask) != 0) {
        Nd += 2*P - (N & mask);
    }
    matrix_init(&U2, Nd, Nd);
    matrix_init(&U1, Nd, Nd);
    matrix_init(&S, Nd, P);

    // U2 = <N>, U1 = <N/2>
    create_rbt_matrix(&U2, &U1, &S, N, Nd, P);
    
    matrix_init(&A0, Nd, Nd);
    matrix_set_values(&A0, unitrand, 0);
    if (N != Nd) 
        clear_extra(&A0, N, 1);

    matrix_init(&A1, Nd, Nd);
    matrix_init(&Ax, Nd, Nd);
    matrix_mcopy(&A1, &A0);

    if (P == 2) {
        // RBT: <N/2>: U1.T*A1;  U1.T*A1*U1
        matrix_mult(&Ax, &U1, &A1, __RSQRT2, 0.0, ARMAS_TRANSA, &conf);
        matrix_mult(&A1, &Ax, &U1, __RSQRT2, 0.0, 0, &conf);
        
        clear_extra(&A1, N, 1);

        // RBT <N>:  U2.T*U1.T*A1*U1;  U2.T*U1.T*A1*U1*U2
        matrix_mult(&Ax, &U2, &A1, __RSQRT2, 0.0, ARMAS_TRANSA, &conf);
        matrix_mult(&A1, &Ax, &U2, __RSQRT2, 0.0, 0, &conf);
    } else if (P == 1) {
        // RBT <N>: U2.T*A1; U2.T*A*U2
        matrix_mult(&Ax, &U2, &A1, __RSQRT2, 0.0, ARMAS_TRANSA, &conf);
        matrix_mult(&A1, &Ax, &U2, __RSQRT2, 0.0, 0, &conf);
    }

    // compute RBT 
    if (N != Nd) {
        matrix_submatrix(&Au, &A0, 0, 0, N, N);
        matrix_submatrix(&Su, &S, 0, 0, N, P);
        matrix_update2_rbt(&Au, &Su, &Su, &conf);
    } else {
        matrix_update2_rbt(&A0, &S, &S, &conf);
    }

    if (verbose > 1 && Nd < 10) {
        matrix_submatrix(&Au, &A0, 0, 0, N, N);
        printf("update_rbt(A0, U)\n"); matrix_printf(stdout, "%6.3f", &Au);
        matrix_submatrix(&Au, &A1, 0, 0, N, N);
        printf("S.T*A0*S\n"); matrix_printf(stdout, "%6.3f", &Au);
    }
    
    matrix_submatrix(&Au, &A1, 0, 0, N, N);
    matrix_submatrix(&Az, &A0, 0, 0, N, N);
    n0 = rel_error(&n1, &Au, &Az, ARMAS_NORM_INF, 0, &conf);
    ok = n0 == 0.0 || isOK(n0, N);
    printf("%4s: rbt(A, U, U) == U.T*A*U [N=%d, Nd=%d]\n", PASS(ok), N, Nd);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    
    matrix_release(&U1);
    matrix_release(&U2);
    matrix_release(&S);
    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&Ax);

    return 1 - ok;
}

#if FLOAT32
#define MAX_ERROR  1e-6
#else
#define MAX_ERROR  1e-12
#endif

int test_rbt_lufactor(int N, int P, int verbose)
{
    __Matrix A0, A1, B0, U, V, B1, X1;
    __Dtype n0, n1;
    armas_pivot_t p;
    int ok, fails = 0;
    armas_conf_t conf = *armas_conf_default();
    
    matrix_init(&A0, N, N);
    matrix_init(&A1, N, N);
    matrix_init(&U, N, P);
    matrix_init(&V, N, P);
    armas_pivot_init(&p, N);
    
    matrix_init(&B0, N, N/2);
    matrix_init(&B1, N, N/2);
    matrix_init(&X1, N, N/2);

    matrix_set_values(&A0, unitrand, 0);
    // make it nearly singular ..
    // ( 2*y  1     ... )      ( 2*y  1 ... )
    // (  y  y/2+e  ... ) -->  ( y/2  e ... )
    // (  ....          )      (  .....     )
    matrix_set(&A0, 0, 0, 2.0);
    matrix_set(&A0, 0, 1, 1.0);
    matrix_set(&A0, 1, 0, 1.0);
    matrix_set(&A0, 1, 1, 0.5+_EPS);
    matrix_mcopy(&A1, &A0);
    matrix_set_values(&B0, unitrand, 0);
    matrix_mcopy(&B1, &B0);
    if (verbose > 2 && N < 10) {
        printf("A:\n"); matrix_printf(stdout, "%6.3f", &A0);
    }
    
    matrix_gen_rbt(&U);
    matrix_gen_rbt(&V);
    // U.T*A*V
    matrix_update2_rbt(&A1, &U, &V, &conf);
    matrix_lufactor(&A1, (armas_pivot_t *)0, &conf);
    // U.T*B
    matrix_mcopy(&X1, &B1);
    matrix_mult_rbt(&X1, &U, ARMAS_LEFT|ARMAS_TRANS, &conf);
    // solve
    matrix_lusolve(&X1, &A1, (armas_pivot_t *)0, 0, &conf);
    // X = V*B
    matrix_mult_rbt(&X1, &V, ARMAS_LEFT, &conf);
    matrix_mult(&B1, &A0, &X1, 1.0, 0.0, 0, &conf);
    n0 = rel_error(&n1, &B1, &B0, ARMAS_NORM_INF, 0, &conf);
    ok = n0 == 0.0 || isFINE(n0, N*MAX_ERROR);
    fails += 1 - ok;
    printf("%4s: || X - V*genp(U.T*A*V, U*B) ||\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error/rbt  || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 2 && N < 10) {
        printf("genp(U.T*A*V):\n"); matrix_printf(stdout, "%13e", &A1);
    }

    // same without randomization....
    matrix_mcopy(&A1, &A0);
    matrix_lufactor(&A1, (armas_pivot_t *)0, &conf);
    matrix_mcopy(&X1, &B0);
    matrix_lusolve(&X1, &A1, (armas_pivot_t *)0, 0, &conf);
    
    matrix_mult(&B1, &A0, &X1, 1.0, 0.0, 0, &conf);
    n0 = rel_error(&n1, &B1, &B0, ARMAS_NORM_INF, 0, &conf);
    if (verbose > 0) {
        printf("   || rel error/genp || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 2 && N < 10) {
        printf("genp(A):\n"); matrix_printf(stdout, "%13e", &A1);
    }

    // same with pivoting....
    matrix_mcopy(&A1, &A0);
    matrix_lufactor(&A1, &p, &conf);
    matrix_mcopy(&X1, &B0);
    matrix_lusolve(&X1, &A1, &p, 0, &conf);
    
    matrix_mult(&B1, &A0, &X1, 1.0, 0.0, 0, &conf);
    n0 = rel_error(&n1, &B1, &B0, ARMAS_NORM_INF, 0, &conf);
    if (verbose > 0) {
        printf("   || rel error/gepp || : %e, [%d]\n", n0, ndigits(n0));
    }
    if (verbose > 2 && N < 10) {
        printf("gepp(A):\n"); matrix_printf(stdout, "%13e", &A1);
    }

    matrix_release(&A0);
    matrix_release(&A1);
    matrix_release(&B0);
    matrix_release(&B1);
    matrix_release(&X1);
    matrix_release(&U);
    matrix_release(&V);

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

    switch (argc-optind) {
    case 2:
        N = atoi(argv[optind]);
        P = atoi(argv[optind+1]);
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
    if (test_update_rbt(N-1, P, verbose))
        fails++;
    if (test_rbt_lufactor(N, P, verbose))
        fails++;
    if (test_rbt_lufactor(N-1, P, verbose))
        fails++;
    return fails;
}


// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
