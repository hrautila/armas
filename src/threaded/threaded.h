
#ifndef ARMAS_THREADED_H
#define ARMAS_THREADED_H 1

#include <stddef.h>
#include "accel.h"

struct armas_threaded_conf {
    size_t max_cores;
    size_t items_per_thread;
};

union blas {
    struct armas_ac_blas1 *blas1;
    struct armas_ac_blas2 *blas2;
    struct armas_ac_blas3 *blas3;
};

struct threaded_block_args {
    int fcol;
    int lcol;
    int frow;
    int lrow;
    int last;
    int index;
    union blas u;
};

extern int armas_ac_threaded_mult(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf);
extern int armas_ac_threaded_mult_sym(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf);
extern int armas_ac_threaded_mult_trm(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf);
extern int armas_ac_threaded_solve_trm(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_threaded_conf *acf);

extern size_t armas_ac_threaded_cores(struct armas_threaded_conf *acf, size_t items);

// Calculate how many row/column blocks are needed with blocking size WB.
static inline
int blocking(int M, int N, int WB, int *nM, int *nN)
{
    *nM = M/WB;
    *nN = N/WB;
    if (M % WB > WB/10) {
        *nM += 1;
    }
    if (N % WB > WB/10) {
        *nN += 1;
    }
    return (*nM)*(*nN);
}

// compute start of k'th out of nblk block when block size wb and total is K
// requires: K/wb == nblk or K/wb == nblk-1
static inline
int block_index(int k, int nblk, int wb, int K)
{
  return k == nblk ? K : k*wb;
}

// compute start of i'th block out of num_blks in nitems elements
static inline
int block_index4(int i, int num_blks, int nitems) {
    if (i == num_blks) {
        return nitems;
    }
    return i*nitems/num_blks - ((i*nitems/num_blks) & 0x3);
}

static inline
int block_index2(int i, int n, int sz) {
    if (i == n) {
        return sz;
    }
    return i*sz/n - ((i*sz/n) & 0x1);
}


#endif // ARMAS_THREADED_H
