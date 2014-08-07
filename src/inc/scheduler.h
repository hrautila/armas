
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_SCHEDULER_H
#define __ARMAS_SCHEDULER_H 1

#define __USE_GNU
#include <sched.h>
#include <pthread.h>

#include "dtype.h"
#include "internal.h"
#include "chan.h"

struct armas_task {
    struct armas_task *next;
    unsigned int id;  // task id
    unsigned int wid; // running worker id
    unsigned int wcnt;
    armas_counter_t *ready;
    void *(*task)(void *);
    void *arg;
};
typedef struct armas_task armas_task_t;
typedef struct armas_task *armas_task_ptr;

static inline
armas_task_t * armas_task_init(armas_task_t *T, unsigned int id,
                               void *(*func)(void *), void *arg, armas_counter_t *c)
{
    T->id = id;
    T->wid = 0;
    T->task = func;
    T->arg = arg;
    T->ready = c;
    T->next = NULL;
    return T;
}

// run a after b
static inline
void armas_task_after(armas_task_t *a, armas_task_t *b)
{
  b->next = a;
}

// task structure for BLAS3 functions
struct blas_task {
  armas_task_t t;
  kernel_param_t kp;
};
typedef struct blas_task blas_task_t;



// define task channel
DefineChannelType(armas_task_ptr, taskq)
// here we have task channel type taskq_t defined.

typedef unsigned long armas_cpuset_t;
#define ARMAS_EMPTY_CPUSET  0

struct armas_scheduler;

typedef struct armas_worker {
    unsigned int id;
    int cpuid;
    int running;
    taskq_t inqueue;
    pthread_t tid;
    int nsched;
    int nexec;
    struct armas_scheduler *sched;
} armas_worker_t;

typedef struct armas_scheduler {
    armas_worker_t *workers;
    cpu_set_t cpus;
    pthread_t tid;
    int opts;
    int nworker;
    int rrindex;
    int status;
    int nsched;
} armas_scheduler_t;

extern void armas_sched_init(armas_scheduler_t *S, int n, int qlen);
extern void armas_sched_stop(armas_scheduler_t *S);
extern void armas_sched_schedule(armas_scheduler_t *S, armas_task_t *T);
extern void armas_schedule(armas_task_t *T);
extern armas_scheduler_t *armas_sched_default(void);
extern int armas_nth_cpu(cpu_set_t *cpus, int n);

#endif  // __ARMAS_SCHEDULER_H

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
 
