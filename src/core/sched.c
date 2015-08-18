
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#define __USE_GNU
#include <sched.h>
#include <pthread.h>

#include <armas/armas.h>
#include "sync.h"
#include "scheduler.h"

//extern void armas_schedule(armas_task_t *T);


static inline
void worker_init(armas_worker_t *W, unsigned long id, int sz, struct armas_scheduler *s)
{
    W->id = id;
    W->cpuid = -1;
    taskq_init(&W->inqueue, sz);
    W->running = 0;
    W->tid = 0;
    W->nsched = 0;
    W->nexec = 0;
    W->sched = s;
}


static 
void *worker_thread(void *arg)
{
    armas_worker_t *W = (armas_worker_t *)arg;
    armas_task_t *T;
    W->running = 1;

    while (1) {
        taskq_read(&W->inqueue, &T);
        if (!T) {
            // null task is sign for stopping;
            break;
        }
        W->nsched++;

        // try to reserve for this worker
        if (cmpxchg(&T->wid, 0, W->id) == 0) {
            // run task
            (T->task)(T->arg);
            // increment ready task counter
            armas_counter_inc(T->ready);
            W->nexec++;
            if (T->next) {
                armas_sched_schedule(W->sched, T->next);
                T->next = 0;
            }
        } else {
            // was already reserved, forget it
        }
        // decrement per task worker count;
        atomic_dec(&T->wcnt);
    }
    W->running = 0;
    return (void *)0;
}

static inline
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

void armas_sched_init(armas_scheduler_t *S, int n, int qlen)
{
    int i, last_cpu;
    S->workers = calloc(n, sizeof(armas_worker_t));
    S->nworker = n;
    last_cpu = -1;

    for (i = 0; i < n; i++) {
        worker_init(&S->workers[i], i+1, qlen, S);
        // pick up next cpu from list available cpus
        last_cpu = next_cpu_in_set(&S->cpus, last_cpu);
        S->workers[i].cpuid = last_cpu;
    }
    S->nsched = 0;
}


void armas_sched_start(armas_scheduler_t *S)
{
    int i;
    for (i = 0; i < S->nworker; i++) {
        worker_start(&S->workers[i]);
    }
    S->status = 1;
}

void armas_sched_stop(armas_scheduler_t *S)
{
    int i;
    for (i = 0; i < S->nworker; i++) {
        taskq_write(&S->workers[i].inqueue, NULL);
    }
    S->status = 0;
}

void armas_sched_schedule(armas_scheduler_t *S, armas_task_t *T)
{
    int k, j;

    if (!S->status) {
        armas_sched_start(S);
    }

    // scheduling ARMAS_SCHED_WORKERS, write directly to worker queues.
    S->nsched++;
    if (S->nworker == 1) {
        T->wcnt = 1;
        taskq_write(&S->workers[0].inqueue, T);
        return;
    }

    if (S->opts & ARMAS_SCHED_TWO) {
        T->wcnt = 2;
        if (S->nworker == 2) {
            taskq_write(&S->workers[0].inqueue, T);
            taskq_write(&S->workers[1].inqueue, T);
            return;
        }
        if ((S->opts & ARMAS_SCHED_ROUNDROBIN) != 0) {
            k = S->rrindex;
            S->rrindex = (S->rrindex + 1) % S->nworker;
            j = S->rrindex;
            S->rrindex = (S->rrindex + 1) % S->nworker;
        } else {
            k = lrand48() % S->nworker;
            for (j = k; j == k; j = lrand48() % S->nworker);
        }
        taskq_write(&S->workers[k].inqueue, T);
        taskq_write(&S->workers[j].inqueue, T);
        return;
    }
    
    T->wcnt = 1;
    if ((S->opts & ARMAS_SCHED_ROUNDROBIN) != 0) {
        k = S->rrindex;
        S->rrindex = (S->rrindex + 1) % S->nworker;
    } else {
        k = lrand48() % S->nworker;
    }
    taskq_write(&S->workers[k].inqueue, T);
}

void armas_schedule(armas_task_t *t)
{
    armas_sched_schedule(armas_sched_default(), t);
}


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
