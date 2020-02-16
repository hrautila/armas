
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int M, int N, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t X, Y, Y0, A, At;
    DTYPE nrm_y, nrm_z;
    int ok;

    armas_x_init(&Y, M, 1);
    armas_x_init(&Y0, M, 1);
    armas_x_init(&X, N, 1);
    armas_x_init(&A, M, N);
    armas_x_init(&At, N, M);

    armas_x_set_values(&Y, zero, ARMAS_NULL);
    armas_x_set_values(&Y0, zero, ARMAS_NULL);
    armas_x_set_values(&X, unitrand, ARMAS_NULL);
    armas_x_set_values(&A, unitrand, ARMAS_NULL);
    armas_x_mcopy(&At, &A, ARMAS_TRANS, cf);

    // Y = A*X
    armas_x_mvmult(0.0, &Y, 1.0, &A, &X, 0, cf);
    nrm_y = armas_x_nrm2(&Y, cf);
    // Y = Y - A^T*X
    armas_x_mvmult(1.0, &Y, -1.0, &At, &X, ARMAS_TRANS, cf);
    if (N < 10 && verbose > 1) {
        printf("Y\n"); armas_x_printf(stdout, "%5.2f", &Y);
    }
    nrm_z = armas_x_nrm2(&Y, cf);
    ok = nrm_z == 0.0 || isOK(nrm_z/nrm_y, N) ? 1 : 0;
    printf("%6s : gemv(A, X) == gemv(A.T, X, T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", nrm_z/nrm_y, ndigits(nrm_z/nrm_y));
    }
    return 1 - ok;
}

int main(int argc, char **argv)
{

    armas_conf_t cf;
    armas_env_t *env;

    int opt, fails = 0;
    int verbose = 1;
    int N = 1307;
    int M = 1025;

    cf = *armas_conf_default();
    env = armas_getenv();

    while ((opt = getopt(argc, argv, "vnr:")) != -1) {
        switch (opt) {
        case 'n':
            cf.optflags |= ARMAS_ONAIVE;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: gemv [-nv -r num M N]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    fails += test_std(M, N, verbose, &cf);

    exit(fails);
}
