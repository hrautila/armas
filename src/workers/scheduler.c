
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#if HAVE_CONFIG_H
#include <config.h>
#endif

// this is only if thread support requested

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#define __USE_GNU
#include <sched.h>
#include <pthread.h>

#include "armas.h"
#include "scheduler.h"
#include "workers.h"

static inline
void worker_init(
    armas_worker_t *W, unsigned long id, int sz, struct armas_scheduler *s)
{
    W->id = id;
    W->cpuid = -1;
    taskq_init(&W->inqueue, sz);
    W->running = 0;
    W->tid = 0;
    W->nsched = 0;
    W->nexec = 0;
    W->sched = s;
    W->cmem = 0;
    W->l1mem = 0;
}

static
void *worker_thread(void *arg)
{
    armas_worker_t *W = (armas_worker_t *)arg;
    struct armas_task *T;

    armas_cbuf_create_thread_global();

    W->running = 1;
    while (W->running) {
        taskq_read(&W->inqueue, &T);
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
            if (T->next) {
                armas_sched_schedule(W->sched, T->next);
                T->next = 0;
            }
            // increment ready task counter
            armas_counter_inc(T->ready);
        } else {
            // was already reserved, decrement and forget it
            atomic_dec(&T->wcnt);
        }
    }
    armas_cbuf_release_thread_global();
    return (void *)0;
}

static
void worker_start(armas_worker_t *W)
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

int armas_sched_conf(armas_scheduler_t *sched, int qlen)
{
    int i, last_cpu;
    struct armas_ac_env *env = armas_ac_getenv();

    sched->workers = calloc(env->max_cores, sizeof(struct armas_worker));
    if (!sched->workers)
        return -ARMAS_EMEMORY;

    sched->nworker = env->max_cores;
    sched->options = env->options;
    last_cpu = -1;

    for (i = 0; i < sched->nworker; i++) {
        worker_init(&sched->workers[i], i+1, qlen, sched);
        // pick up next cpu from list available cpus
        last_cpu = next_cpu_in_set(&sched->cpus, last_cpu);
        sched->workers[i].cpuid = last_cpu;
    }
    sched->nsched = 0;
    return 0;
}

void armas_sched_start(armas_scheduler_t *sched)
{
    int i;
    for (i = 0; i < sched->nworker; i++) {
        worker_start(&sched->workers[i]);
    }
    sched->status = 1;
}

void armas_sched_stop(armas_scheduler_t *sched)
{
    int i;
    for (i = 0; i < sched->nworker; i++) {
        sched->workers[i].running = 0;
        taskq_write(&sched->workers[i].inqueue, NULL);
    }
    for (i = 0; i < sched->nworker; i++) {
        pthread_join(sched->workers[i].tid, (void **)0);
    }
    sched->status = 0;
}

void armas_sched_release(struct armas_scheduler *sched)
{
    if (!sched || sched->status)
        return;
    if (sched->workers) {
        free(sched->workers);
        sched->workers = (struct armas_worker *)0;
    }
    return;
}

void armas_sched_schedule(armas_scheduler_t *sched, struct armas_task *task)
{
    int k, j;

    if (!sched->status) {
        armas_sched_start(sched);
    }

    // scheduling ARMAS_SCHED_WORKERS, write directly to worker queues.
    sched->nsched++;
    if (sched->nworker == 1) {
        task->wcnt = 1;
        taskq_write(&sched->workers[0].inqueue, task);
        return;
    }

    if (sched->options & ARMAS_OSCHED_TWO) {
        task->wcnt = 2;
        if (sched->nworker == 2) {
            taskq_write(&sched->workers[0].inqueue, task);
            taskq_write(&sched->workers[1].inqueue, task);
            return;
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
        return;
    }

    task->wcnt = 1;
    if ((sched->options & ARMAS_OSCHED_ROUNDROBIN) != 0) {
        k = sched->rrindex;
        sched->rrindex = (sched->rrindex + 1) % sched->nworker;
    } else {
        k = lrand48() % sched->nworker;
    }
    taskq_write(&sched->workers[k].inqueue, task);
}
#if 0
void armas_schedule(struct armas_task *task)
{
    armas_sched_schedule(armas_sched_default(), task);
}
#endif
