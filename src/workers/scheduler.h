
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_SCHEDULER_H
#define ARMAS_SCHEDULER_H 1

#include <stddef.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>

// #include "sync.h"

struct armas_ac_counter;
struct armas_ac_scheduler;

struct armas_ac_task {
    unsigned int id;  // task id
    unsigned int wid; // running worker id
    unsigned int wcnt;
    void *(*func)(void *);
    void *args;
    struct armas_ac_counter *ready;
};

static inline
void armas_task_init(struct armas_ac_task *task, unsigned int id,
                     void *(*func)(void *), void *args, struct armas_ac_counter *ready)
{
    task->id = id;
    task->wid = 0;
    task->func = func;
    task->args = args;
    task->ready = ready;
}

struct armas_ac_scheduler_ops {
    int (*start)(struct armas_ac_scheduler *sc);
    int (*stop)(struct armas_ac_scheduler *sc);
    int (*release)(struct armas_ac_scheduler *sc);
    int (*schedule)(struct armas_ac_scheduler *sc, void *task);
};

struct armas_ac_scheduler {
    struct armas_ac_scheduler_ops *vptr;
};

static inline
int armas_sched_start(struct armas_ac_scheduler *sc)
{
    if (!sc || !sc->vptr->start)
        return -1;
    return (sc->vptr->start)(sc);
}

static inline
int armas_sched_stop(struct armas_ac_scheduler *sc)
{
    if (!sc || !sc->vptr->stop)
        return -1;
    return (sc->vptr->stop)(sc);
}

static inline
void armas_sched_release(struct armas_ac_scheduler *sc)
{
    if (!sc || !sc->vptr->release)
        return;
    (sc->vptr->start)(sc);
}

static inline
int armas_sched_schedule(struct armas_ac_scheduler *sc, void *task)
{
    if (!sc || !sc->vptr->schedule)
        return -1;
    return (sc->vptr->schedule)(sc, task);
}


#endif  /* ARMAS_SCHEDULER_H */
