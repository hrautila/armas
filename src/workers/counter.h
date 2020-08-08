
#ifndef ARMAS_COUNTER_H
#define ARMAS_COUNTER_H 1

#include <stddef.h>
#include <assert.h>
#include <pthread.h>
#include "accel.h"


struct armas_ac_counter {
    int count;
    pthread_mutex_t lock;
    pthread_cond_t ready;
};

static inline
void armas_counter_init(struct armas_ac_counter *c, int count)
{
    c->count = count;
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->ready, NULL);
}

static inline
void armas_counter_decrement(struct armas_ac_counter *c)
{
    pthread_mutex_lock(&c->lock);
    c->count--;
    if (c->count == 0) {
        pthread_cond_signal(&c->ready);
    }
    pthread_mutex_unlock(&c->lock);
}

static inline
void armas_counter_wait(struct armas_ac_counter *c)
{
    pthread_mutex_lock(&c->lock);
    if (c->count != 0) {
        pthread_cond_wait(&c->ready, &c->lock);
    }
    pthread_mutex_unlock(&c->lock);
}

struct armas_ac_barrier {
    int count;
    int current;
    int increment;
    pthread_mutex_t lock;
    pthread_cond_t ready;
};

static inline
void armas_barrier_init(struct armas_ac_barrier *b, int nthreads)
{
    b->count = nthreads;
    b->current = 0;
    b->increment = 0;
    pthread_mutex_init(&b->lock, NULL);
    pthread_cond_init(&b->ready, NULL);
}

static inline
void armas_barrier_synchronize(struct armas_ac_barrier *b)
{
    pthread_mutex_lock(&b->lock);
    /* First entry initializes direction */
    if (b->current == 0) {
        b->increment = 1;
    } else if (b->current == b->count) {
        b->increment = -1;
    }
    assert(b->increment == 1 || b->increment == -1);
    b->current += b->increment;
    if (b->current != 0 && b->current != b->count) {
        pthread_cond_wait(&b->ready, &b->lock);
    } else {
        pthread_cond_broadcast(&b->ready);
    }
    pthread_mutex_unlock(&b->lock);
}

#endif /* ARMAS_COUNTER_H */
