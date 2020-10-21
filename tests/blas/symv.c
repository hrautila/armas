
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_dense_t X, Y0, A0;
    const char *uplo = (flags & ARMAS_LOWER) ? "lower" : "upper";
    DTYPE nrm_y, nrm_z;
    int ok;

    armas_init(&Y0, N, 1);
    armas_init(&X, N, 1);
    armas_init(&A0, N, N);

    armas_set_values(&Y0, zero, ARMAS_NULL);
    armas_set_values(&X, unitrand, ARMAS_NULL);
    armas_set_values(&A0, unitrand, ARMAS_SYMM);

    // Y = A*X
    armas_mvmult_sym(0.0, &Y0, 1.0, &A0, &X, flags, cf);
    nrm_y = armas_nrm2(&Y0, cf);
    // Y = Y - A*X
    armas_mvmult(1.0, &Y0, -1.0, &A0, &X, 0, cf);
    nrm_z = armas_nrm2(&Y0, cf);
    ok = nrm_z == 0.0 || isOK(nrm_z/nrm_y, N) ? 1 : 0;
    printf("%6s : %s.symv(A, X) == %s(gemv(A, X))\n", PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", 
            nrm_z/nrm_y, ndigits(nrm_z/nrm_y));
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
    int all = 1;
    int lower = 0;
    int upper = 0;

    cf = *armas_conf_default();
    env = armas_getenv();

    while ((opt = getopt(argc, argv, "vnr:LU")) != -1) {
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
       case 'L':
            lower = 1;
            all = 0;
            break;
        case 'U':
            upper = 1;
            all = 0;
            break;
         default:
            fprintf(stderr, "usage: gemv [-nv -r num M N]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    if (all) {
       fails += test_std(N, verbose, ARMAS_LOWER, &cf);
       fails += test_std(N, verbose, ARMAS_UPPER, &cf);
    } else {
        if (lower)
          fails += test_std(N, verbose, ARMAS_LOWER, &cf);
        if (upper)
           fails += test_std(N, verbose, ARMAS_UPPER, &cf);
    }
    exit(fails);
}
