
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

armas_d_sparse_t *generate(int m, int n, int nz)
{
    armas_d_sparse_t *A;

    A = armassp_d_new(m, n, nz, ARMASSP_COO);
    if (m > n) {
        // on very column need entry
        for (int j = 0; j < n; j++) {
        }
    } else if (m < n) {
        // on every row;
        for (int i = 0; i < m; i++) {
        }
    } else { // m == n
    }     
}

int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = "test1.mtx";
    FILE *fp;
    int tc;
    armas_d_sparse_t *A;

    int uplo = 0;
    int m = 9;
    int n = 9;
    int nz = m*n/5;
    
    while ((opt = getopt(argc, argv, "vf:m:n:N:LU")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'f':
            path = optarg;
            break;
        case 'm':
            m = atoi(optarg);
            break;
        case 'n':
            n = atoi(optarg);
            break;
        case 'N':
            nz = atoi(optarg);
            break;
        case 'L':
            uplo = ARMAS_LOWER;
            break;
        case 'U':
            uplo = ARMAS_UPPER;
            break;
        default:
            fprintf(stderr, "usage: gensp [-vLU -f path -m M -n N -N nz] \n");
            exit(1);
        }
    }
    
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
