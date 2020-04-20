
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

#include "sync.h"

struct armas_counter;

struct armas_task {
    struct armas_task *next;
    unsigned int id;  // task id
    unsigned int wid; // running worker id
    unsigned int wcnt;
    struct armas_counter *ready;
    void *(*func)(void *);
    void *args;
};

static inline
void armas_task_init(struct armas_task *task, unsigned int id,
                     void *(*func)(void *), void *args, struct armas_counter *ready)
{
    task->id = id;
    task->wid = 0;
    task->func = func;
    task->args = args;
    task->ready = ready;
    task->next = (struct armas_task *)0;
}

struct armas_taskq {
    struct armas_task **elems;
    unsigned int size;
    unsigned int count;
    unsigned int head;
    unsigned int tail;
    mutex_t mlock;
};

static inline
void taskq_init(struct armas_taskq *ch, unsigned int qlen)
{
    ch->elems = malloc(qlen * sizeof(struct armas_task *));
    ch->size = qlen;
    ch->count = 0;
    ch->head = 0;
    ch->tail = 0;
    mutex_init(&ch->mlock, NULL);
}

static inline
void taskq_release(struct armas_taskq *ch)
{
    if (ch && ch->elems)
        free(ch->elems);
}

static inline
struct armas_taskq *taskq_create(unsigned int qlen)
{
    struct armas_taskq *ch = malloc(sizeof(struct armas_taskq));
    if (ch)
        taskq_init(ch, qlen);
    return ch;
}

static inline
void taskq_free(struct armas_taskq *ch)
{
    if (ch) {
        taskq_release(ch);
        free(ch);
    }
}

static inline
int taskq_enqueue(struct armas_taskq *ch, struct armas_task *v)
{
    mutex_lock(&ch->mlock);
    if (ch->count == ch->size) {
        mutex_unlock(&ch->mlock);
        return -1;
    }
    int signal = ch->count == 0;
    ch->elems[ch->tail] = v;
    ch->count++;
    ch->tail = (ch->tail + 1) % ch->size;
    if (signal)
        futex_wake(&ch->head, 1);
    mutex_unlock(&ch->mlock);
    return 0;
}


static inline
int taskq_dequeue(struct armas_taskq *ch, struct armas_task **v)
{
    mutex_lock(&ch->mlock);
    if (ch->count == 0) {
        mutex_unlock(&ch->mlock);
        return -1;
    }
    *v = ch->elems[ch->head];
    int signal = ch->count == ch->size;
    ch->count--;
    ch->head = (ch->head + 1) % ch->size;
    if (signal)
        futex_wake(&ch->tail, 1);
    mutex_unlock(&ch->mlock);
    return 0;
}

// empty: head == tail && count == 0
//   before sleeping try to wake up some writer

static inline
void taskq_wait_onempty(struct armas_taskq *ch, unsigned int usec)
{
    int i;
    volatile unsigned int *cntr = &ch->count;
    struct timespec *tp, tm = {0, usec * 1000};
    futex_wake(&ch->tail, 1);
    for (i = 0; i < 100; i++) {
        if (*cntr != 0)
            return;
        cpu_relax();
    }
    tp = usec == 0 ? NULL : &tm;
    futex_wait(&ch->head, ch->tail, tp);
}

// full: tail == head && count == size
//   before sleeping try to wake up some reader
static inline
void taskq_wait_onfull(struct armas_taskq *ch, unsigned int usec)
{
    int i;
    volatile unsigned int *cntr = &ch->count;
    struct timespec *tp, tm = {0, 100 * 1000};
    futex_wake(&ch->head, 1);
    for (i = 0; i < 100; i++) {
        if (*cntr != ch->size)
            return;
        cpu_relax();
    }
    tp = usec == 0 ? NULL : &tm;
    futex_wait(&ch->tail, ch->head, tp);
}


static inline
void taskq_read(struct armas_taskq *ch, struct armas_task **v)
{
    while (taskq_dequeue(ch, v) != 0) {
        taskq_wait_onempty(ch, 100);
    }
}


static inline
void taskq_write(struct armas_taskq *ch, struct armas_task *v)
{
    while (taskq_enqueue(ch, v) != 0) {
        taskq_wait_onfull(ch, 100);
    }
}

struct armas_scheduler;

typedef struct armas_worker {
    unsigned int id;
    int cpuid;
    int running;
    struct armas_taskq inqueue;
    pthread_t tid;
    int nsched;
    int nexec;
    size_t cmem;
    size_t l1mem;
    struct armas_scheduler *sched;
} armas_worker_t;

typedef struct armas_scheduler {
    struct armas_worker *workers;
    cpu_set_t cpus;
    pthread_t tid;
    unsigned int options;
    unsigned int nworker;
    int rrindex;
    unsigned int status;
    unsigned int nsched;
} armas_scheduler_t;

struct armas_workers_env;

extern void armas_sched_init(armas_scheduler_t *S, int n, int qlen);
extern int armas_sched_conf(armas_scheduler_t *S, int qlen);
extern void armas_sched_stop(armas_scheduler_t *S);
extern void armas_sched_schedule(armas_scheduler_t *S, struct armas_task *T);
extern void armas_schedule(struct armas_task*T);
extern struct armas_scheduler *armas_sched_default(void);
extern int armas_nth_cpu(cpu_set_t *cpus, int n);

#endif  /* ARMAS_SCHEDULER_H */
