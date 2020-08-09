
#ifndef ARMAS_QUEUE_H
#define ARMAS_QUEUE_H 1

#include <stddef.h>
#include <pthread.h>
#include "accel.h"

struct armas_ac_queue {
    void **items;
    uint32_t size;
    uint32_t count;
    uint32_t head;
    uint32_t tail;
    pthread_mutex_t mlock;
    pthread_cond_t empty;
    pthread_cond_t full;
};
typedef struct armas_ac_queue armas_ac_queue_t;

static inline
int taskq_init(struct armas_ac_queue *ch, unsigned int qlen)
{
    ch->items = malloc(qlen * sizeof(void *));
    if (!ch->items)
        return -1;
    ch->size = qlen;
    ch->count = 0;
    ch->head = 0;
    ch->tail = 0;
    pthread_mutex_init(&ch->mlock, NULL);
    pthread_cond_init(&ch->empty, NULL);
    pthread_cond_init(&ch->full, NULL);
    return 0;
}

static inline
void taskq_release(struct armas_ac_queue *ch)
{
    if (ch && ch->items)
        free(ch->items);
}

static inline
struct armas_ac_queue *taskq_create(unsigned int qlen)
{
    struct armas_ac_queue *ch = malloc(sizeof(struct armas_ac_queue));
    if (ch)
        taskq_init(ch, qlen);
    return ch;
}

static inline
void taskq_free(struct armas_ac_queue *ch)
{
    if (ch) {
        taskq_release(ch);
        free(ch);
    }
}

static inline
int taskq_append(struct armas_ac_queue *ch, void *v)
{
    int signal = ch->count == 0;
    ch->items[ch->tail] = v;
    ch->count++;
    ch->tail = (ch->tail + 1) % ch->size;
    return signal;
}

static inline
int taskq_remove(struct armas_ac_queue *ch, void **v)
{
    int signal = ch->count == ch->size;
    *v = ch->items[ch->head];
    ch->items[ch->head] = (void *)0;
    ch->count--;
    ch->head = (ch->head + 1) % ch->size;
    return signal;
}

static inline
int taskq_enqueue_nowait(struct armas_ac_queue *ch, void *v, int broadcast)
{
    pthread_mutex_lock(&ch->mlock);
    if (ch->count == ch->size) {
        pthread_mutex_unlock(&ch->mlock);
        return -1;
    }
    int signal = taskq_append(ch, v);
    if (signal) {
        if (broadcast)
            pthread_cond_broadcast(&ch->empty);
        else
            pthread_cond_signal(&ch->empty);
    }
    pthread_mutex_unlock(&ch->mlock);
    return 0;
}

static inline
int taskq_enqueue(struct armas_ac_queue *ch, void *v, int broadcast)
{
    pthread_mutex_lock(&ch->mlock);
    while (ch->count == ch->size) {
        pthread_cond_wait(&ch->full, &ch->mlock);
    }
    int signal = taskq_append(ch, v);
    if (signal) {
        if (broadcast)
            pthread_cond_broadcast(&ch->empty);
        else
            pthread_cond_signal(&ch->empty);
    }
    pthread_mutex_unlock(&ch->mlock);
    return 0;
}


static inline
int taskq_dequeue_nowait(struct armas_ac_queue *ch, void **v, int broadcast)
{
    pthread_mutex_lock(&ch->mlock);
    if (ch->count == 0) {
        pthread_mutex_unlock(&ch->mlock);
        return -1;
    }
    int signal = taskq_remove(ch, v);
    if (signal) {
        if (broadcast)
            pthread_cond_broadcast(&ch->empty);
        else
            pthread_cond_signal(&ch->empty);
    }
    pthread_mutex_unlock(&ch->mlock);
    return 0;
}

static inline
int taskq_dequeue(struct armas_ac_queue *ch, void **v, int broadcast)
{
    pthread_mutex_lock(&ch->mlock);
    while (ch->count == 0) {
        pthread_cond_wait(&ch->empty, &ch->mlock);
    }
    int signal = taskq_remove(ch, v);
    if (signal) {
        if (broadcast)
            pthread_cond_broadcast(&ch->empty);
        else
            pthread_cond_signal(&ch->empty);
    }
    pthread_mutex_unlock(&ch->mlock);
    return 0;
}

// empty: head == tail && count == 0
//   before sleeping try to wake up some writer

static inline
int taskq_wait_until_empty(struct armas_ac_queue *ch, unsigned int usec)
{
    int rc = 0;
    struct timespec tm = {0, usec * 1000};
    pthread_mutex_lock(&ch->mlock);
    if (ch->count != 0) {
        rc = pthread_cond_timedwait(&ch->empty, &ch->mlock, &tm);
    }
    pthread_mutex_unlock(&ch->mlock);
    return rc;
}

static inline
int taskq_wait_onempty(struct armas_ac_queue *ch, unsigned int usec)
{
    int rc = 0;
    struct timespec tm = {0, usec * 1000};
    pthread_mutex_lock(&ch->mlock);
    if (ch->count == 0) {
        rc = pthread_cond_timedwait(&ch->empty, &ch->mlock, &tm);
    }
    pthread_mutex_unlock(&ch->mlock);
    return rc;
}

static inline
void taskq_wait_while_empty(struct armas_ac_queue *ch)
{
    pthread_mutex_lock(&ch->mlock);
    if (ch->count == 0) {
        pthread_cond_wait(&ch->empty, &ch->mlock);
    }
    pthread_mutex_unlock(&ch->mlock);
}

// full: tail == head && count == size
//   before sleeping try to wake up some reader
static inline
int taskq_wait_onfull(struct armas_ac_queue *ch, unsigned int usec)
{
    int rc = 0;
    struct timespec tm = {0, usec * 1000};
    pthread_mutex_lock(&ch->mlock);
    if (ch->count == ch->size) {
        rc = pthread_cond_timedwait(&ch->full, &ch->mlock, &tm);
    }
    pthread_mutex_unlock(&ch->mlock);
    return rc;
}

static inline
void taskq_wait_while_full(struct armas_ac_queue *ch)
{
    pthread_mutex_lock(&ch->mlock);
    if (ch->count == ch->size) {
        pthread_cond_wait(&ch->empty, &ch->mlock);
    }
    pthread_mutex_unlock(&ch->mlock);
}

static inline
void taskq_read(struct armas_ac_queue *ch, void **v)
{
    while (taskq_dequeue_nowait(ch, v, 1) != 0) {
        taskq_wait_onempty(ch, 100);
    }
}


static inline
void taskq_write(struct armas_ac_queue *ch, void *v)
{
    while (taskq_enqueue_nowait(ch, v, 1) != 0) {
        taskq_wait_onfull(ch, 100);
    }
}

#endif /* ARMAS_QUEUE_H */
