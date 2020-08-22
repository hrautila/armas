
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
#define atomic_dec(P) __sync_sub_and_fetch((P), 1)
#define atomic_inc(P) __sync_add_and_fetch((P), 1)

struct armas_ac_sched_transient;

struct armas_ac_thread {
    pthread_t tid;
    struct armas_ac_sched_transient *sched;
    int running;
};

struct armas_ac_sched_transient {
    struct armas_ac_scheduler sched;
    struct armas_ac_queue task_queue;
    struct armas_ac_thread *threads;
    unsigned int options;
    size_t nworker;
    size_t nrunning;
    unsigned int status;
};

static
void *transient_thread(void *arg)
{
    struct armas_ac_task *T;
    struct armas_ac_thread *self = (struct armas_ac_thread *)arg;
    struct armas_ac_sched_transient *sc = self->sched;

    armas_cbuf_create_thread_global();

    while (1) {
        if (taskq_dequeue_nowait(&sc->task_queue, (void **)&T, 1) < 0)
            break;

        (T->func)(T->args);
        armas_counter_decrement(T->ready);
    }

    armas_cbuf_release_thread_global();

    cmpxchg(&self->running, 1, 0);
    atomic_dec(&sc->nrunning);
    return (void *)0;
}

static
int sched_transient_start(struct armas_ac_scheduler *sc)
{
    struct armas_ac_sched_transient *sched = (struct armas_ac_sched_transient *)sc;
    sched->status = 1;
    return 0;
}

static
int sched_transient_stop(struct armas_ac_scheduler *sc)
{
    return 0;
}

static
int sched_transient_release(struct armas_ac_scheduler *sc)
{
    struct armas_ac_sched_transient *sched = (struct armas_ac_sched_transient *)sc;
    if (!sched)
        return -ARMAS_EINVAL;

    taskq_release(&sched->task_queue);
    free(sched);
    return 0;
}

static
int sched_transient_schedule(struct armas_ac_scheduler *sc, void *t)
{
    unsigned int k;
    struct armas_ac_sched_transient *sched = (struct armas_ac_sched_transient *)sc;
    struct armas_ac_task *task = (struct armas_ac_task *)t;

    taskq_enqueue(&sched->task_queue, task, 1);
    if (sched->nrunning < sched->nworker) {
        for (k = 0; k < sched->nworker && sched->threads[k].running; k++);

        sched->threads[k].running = 1;
        sched->threads[k].sched = sched;
        pthread_create(&sched->threads[k].tid, NULL, transient_thread, &sched->threads[k]);
        atomic_inc(&sched->nrunning);
    }
    return 0;
}

static struct armas_ac_scheduler_ops ops = {
    .start =  sched_transient_start,
    .stop = sched_transient_stop,
    .release = sched_transient_release,
    .schedule = sched_transient_schedule
};

int armas_ac_sched_transient_init(struct armas_ac_scheduler **scheduler, int qlen)
{
    unsigned char *buf;
    struct armas_ac_env *env = armas_ac_getenv();

    size_t nbytes = sizeof(struct armas_ac_sched_transient) + env->max_cores * sizeof(struct armas_ac_thread);

    struct armas_ac_sched_transient *sc = calloc(1, nbytes);
    if (!sc)
        return -ARMAS_EMEMORY;

    if (taskq_init(&sc->task_queue, 2*env->max_cores) < 0) {
        free(sc);
        return -ARMAS_EMEMORY;
    }
    buf = (unsigned char *)sc;
    sc->threads = (struct armas_ac_thread *)&buf[sizeof(struct armas_ac_sched_transient)];
    sc->nworker = env->max_cores;
    sc->nrunning = 0;
    sc->options = env->options;

    sc->sched.vptr = &ops;
    *scheduler = (struct armas_ac_scheduler *)sc;
    return 0;
}
