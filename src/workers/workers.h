
#ifndef ARMAS_WORKERS_H
#define ARMAS_WORKERS_H 1

#include <stddef.h>
#define __USE_GNU
#include <sched.h>
#include "accel.h"
#include "scheduler.h"
//#include "task.h"

#define DEFAULT_QLEN    16

// task structure
struct armas_ac_worker_task {
    struct armas_ac_task task;
    struct armas_ac_block args;
};

struct armas_ac_workers {
    struct armas_ac_scheduler *sched;
};

extern int armas_ac_workers_mult(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_ac_workers *wcf);
extern int armas_ac_workers_mult_sym(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_ac_workers *wcf);
extern int armas_ac_workers_mult_trm(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_ac_workers *wcf);
extern int armas_ac_workers_solve_trm(struct armas_ac_blas3 *args, armas_conf_t *cf, struct armas_ac_workers *wcf);

extern int armas_ac_sched_workers_init(struct armas_ac_scheduler **scheduler, int qlen);

// extern void armas_ac_workers_env(struct armas_workers_env *env);

static inline
int worker_tiles(int *nrow, int *ncol, int M, int N, int wb)
{
    int qr = M % wb > wb/8;
    int qc = N % wb > wb/8;
    *nrow = M < wb ? 1 : M/wb + qr;
    *ncol = N < wb ? 1 : N/wb + qc;
    return (*nrow)*(*ncol);
}

static inline
int worker_all_ready(struct armas_ac_worker_task *tasks, int ntask)
{
    int refcount;
    for (int k = 0; k < 50; k++) {
        refcount = 0;
        for (int i = 0; i < ntask; i++) {
            refcount += tasks[i].task.wcnt;
        }
        if (refcount == 0)
            break;
    }
    return refcount;
}

#endif /* ARMAS_WORKERS_H */
