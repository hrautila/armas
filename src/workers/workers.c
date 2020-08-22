
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*
 *  Here is implementation of a scheduler with static, persistent worker threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#define __USE_GNU
#include <sched.h>
#include <pthread.h>

#include "armas.h"
#include "queue.h"
#include "counter.h"
#include "scheduler.h"
#include "workers.h"


#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))
// atomically decrement/increment value at memory location P and return new value
#define atomic_dec(P) __sync_sub_and_fetch((P), 1)

typedef struct armas_ac_worker {
    struct armas_ac_scheduler *sched;
    struct armas_ac_queue inqueue;
    unsigned int id;
    int cpuid;
    int running;
    pthread_t tid;
    int nsched;
    int nexec;
    size_t cmem;
    size_t l1mem;
} armas_ac_worker_t;

typedef struct armas_ac_sched_workers {
    struct armas_ac_scheduler sched;
    struct armas_ac_worker *workers;
    cpu_set_t cpus;
    pthread_t tid;
    unsigned int options;
    unsigned int nworker;
    unsigned int rrindex;
    unsigned int status;
    unsigned int nsched;
} armas_scheduler_t;

static inline
void worker_init(
    armas_ac_worker_t *W, unsigned long id, unsigned int sz, struct armas_ac_sched_workers *sched)
{
    W->id = id;
    W->cpuid = -1;
    taskq_init(&W->inqueue, sz);
    W->running = 0;
    W->tid = 0;
    W->nsched = 0;
    W->nexec = 0;
    W->sched = (struct armas_ac_scheduler *)sched;
    W->cmem = 0;
    W->l1mem = 0;
}

static
void *worker_thread(void *arg)
{
    armas_ac_worker_t *W = (armas_ac_worker_t *)arg;
    struct armas_ac_task *T;

    armas_cbuf_create_thread_global();

    W->running = 1;
    while (W->running) {
        taskq_read(&W->inqueue, (void **)&T);
        if (!T) {
            continue;
        }
        W->nsched++;

        // try to reserve for this worker
        if (cmpxchg(&T->wid, 0, W->id) == 0) {
            // decrement per task worker count;
            atomic_dec(&T->wcnt);
            // run task
            (T->func)(T->args);
            W->nexec++;
            // decrement task counter
            armas_counter_decrement(T->ready);
        } else {
            // was already reserved, decrement and forget it
            atomic_dec(&T->wcnt);
        }
    }
    armas_cbuf_release_thread_global();
    return (void *)0;
}

static
void worker_start(struct armas_ac_worker *W)
{
    cpu_set_t cpuset;
    pthread_create(&W->tid, NULL, worker_thread, W);
    if (W->cpuid != -1) {
        // set cpu affinity
        CPU_ZERO(&cpuset);
        CPU_SET(W->cpuid, &cpuset);
        //printf(".. set worker %d affinity to cpu %d\n", W->id, W->cpuid);
        pthread_setaffinity_np(W->tid, sizeof(cpuset), &cpuset);
    }
    return;
}

static inline
int next_cpu_in_set(cpu_set_t *cpus, int last)
{
    int n;
    for (n = last+1; n < CPU_SETSIZE; n++) {
        if (CPU_ISSET(n, cpus)) {
            return n;
        }
    }
    return -1;
}

/*
 * Get CPU number of n'th CPU in cpu_set. N as values 1 ... CPU_SETSIZE.
 */
int armas_nth_cpu(cpu_set_t *cpus, int n)
{
    int k;
    for (k = 0; k < CPU_SETSIZE; k++) {
        if (CPU_ISSET(k, cpus))
            n--;
        if (n == 0)
            break;
    }
    return k == CPU_SETSIZE ? -1 : k;
}

static
int sched_workers_start(struct armas_ac_scheduler *sc)
{
    struct armas_ac_sched_workers *sched = (struct armas_ac_sched_workers *)sc;
    for (int i = 0; i < sched->nworker; i++) {
        worker_start(&sched->workers[i]);
    }
    sched->status = 1;
    return 0;
}

static
int sched_workers_stop(struct armas_ac_scheduler *sc)
{
    struct armas_ac_sched_workers *sched = (struct armas_ac_sched_workers *)sc;
    for (int i = 0; i < sched->nworker; i++) {
        sched->workers[i].running = 0;
        taskq_write(&sched->workers[i].inqueue, NULL);
    }
    for (int i = 0; i < sched->nworker; i++) {
        pthread_join(sched->workers[i].tid, (void **)0);
    }
    sched->status = 0;
    return 0;
}

static
int sched_workers_release(struct armas_ac_scheduler *sc)
{
    struct armas_ac_sched_workers *sched = (struct armas_ac_sched_workers *)sc;
    if (!sched || sched->status)
        return -1;
    free(sched);
    return 0;
}

static
int sched_workers_schedule(struct armas_ac_scheduler *sc, void *t)
{
    unsigned int k, j;
    struct armas_ac_sched_workers *sched = (struct armas_ac_sched_workers *)sc;
    struct armas_ac_task *task = (struct armas_ac_task *)t;

    if (!sched->status) {
        armas_sched_start(sc);
    }

    // scheduling ARMAS_SCHED_WORKERS, write directly to worker queues.
    sched->nsched++;
    if (sched->nworker == 1) {
        task->wcnt = 1;
        taskq_write(&sched->workers[0].inqueue, task);
        return 0;
    }

    if (sched->options & ARMAS_OSCHED_TWO) {
        task->wcnt = 2;
        if (sched->nworker == 2) {
            taskq_write(&sched->workers[0].inqueue, task);
            taskq_write(&sched->workers[1].inqueue, task);
            return 0;
        }
        if ((sched->options & ARMAS_OSCHED_ROUNDROBIN) != 0) {
            k = sched->rrindex;
            sched->rrindex = (sched->rrindex + 1) % sched->nworker;
            j = sched->rrindex;
            sched->rrindex = (sched->rrindex + 1) % sched->nworker;
        } else {
            k = lrand48() % sched->nworker;
            for (j = k; j == k; j = lrand48() % sched->nworker);
        }
        taskq_write(&sched->workers[k].inqueue, task);
        taskq_write(&sched->workers[j].inqueue, task);
        return 0;
    }

    task->wcnt = 1;
    if ((sched->options & ARMAS_OSCHED_ROUNDROBIN) != 0) {
        k = sched->rrindex;
        sched->rrindex = (sched->rrindex + 1) % sched->nworker;
    } else {
        k = lrand48() % sched->nworker;
    }
    taskq_write(&sched->workers[k].inqueue, task);
    return 0;
}

static struct armas_ac_scheduler_ops ops = {
    .start =  sched_workers_start,
    .stop = sched_workers_stop,
    .release = sched_workers_release,
    .schedule = sched_workers_schedule
};

int armas_ac_sched_workers_init(struct armas_ac_scheduler **scheduler, int qlen)
{
    int last_cpu;
    unsigned char *buf;
    struct armas_ac_env *env = armas_ac_getenv();

    size_t nbytes = sizeof(struct armas_ac_sched_workers) +
         env->max_cores * sizeof(struct armas_ac_worker);

    struct armas_ac_sched_workers *sc = calloc(1, nbytes);
    if (!sc)
        return -ARMAS_EMEMORY;

    buf = (unsigned char *)sc;
    sc->workers = (struct armas_ac_worker *)&buf[sizeof(struct armas_ac_sched_workers)];
    sc->nworker = env->max_cores;
    sc->options = env->options;
    last_cpu = -1;

    for (int i = 0; i < sc->nworker; i++) {
        worker_init(&sc->workers[i], i+1, qlen, sc);
        // pick up next cpu from list available cpus
        last_cpu = next_cpu_in_set(&sc->cpus, last_cpu);
        sc->workers[i].cpuid = last_cpu;
    }
    sc->nsched = 0;

    sc->sched.vptr = &ops;
    *scheduler = (struct armas_ac_scheduler *)sc;
    return 0;
}
