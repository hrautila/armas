
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_SYNC_H
#define ARMAS_SYNC_H 1

#include <linux/futex.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>

#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))
#define xchg64(P, V) __atomic_exchange_8((P), (V), __ATOMIC_SEQ_CST)
#define xchg32(P, V) __atomic_exchange_4((P), (V), __ATOMIC_SEQ_CST)
#define xchg16(P, V) __atomic_exchange_2((P), (V), __ATOMIC_SEQ_CST)

// atomically decrement/increment value at memory location P and return new value
#define atomic_dec(P) __sync_sub_and_fetch((P), 1)
#define atomic_inc(P) __sync_add_and_fetch((P), 1) 

#ifdef __x86_64__
#define cpu_relax() asm volatile("pause\n": : :"memory")
#else
#define cpu_relax() asm volatile("": : :"memory")
#endif

static inline
int futex_wait(volatile void *addr, int val, struct timespec *tm)
{
    return syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, val, tm, NULL, NULL);
}

static inline
int futex_wake(volatile void *addr, int cnt)
{
    return syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, cnt, NULL, NULL, NULL);
}

// Mutex implementation as descriped in
//  (1) Ulrich Drepper, Mutexes are Tricky, 2011/11/5 v1.6
//
typedef struct mutex {
    int val;
} mutex_t;

static inline
void mutex_init(mutex_t *m, void *attr)
{
    m->val = 0;
}

static inline
void mutex_lock(mutex_t *m)
{
    int c;
    if ((c = cmpxchg(&m->val, 0, 1)) != 0) {
        if (c != 2)
            c = xchg32(&m->val, 2);
        while (c != 0) {
            futex_wait(&m->val, 2, NULL);
            c = xchg32(&m->val, 2);
        }
    }
}

static inline
void mutex_unlock(mutex_t *m)
{
    if (__sync_fetch_and_sub(&m->val, 1) != 1) {
        m->val = 0;
        futex_wake(&m->val, 1);
    }
}

// waitable counter
typedef struct armas_counter
{
    int target; 
    int count;
} armas_counter_t;

static inline
void armas_counter_init(armas_counter_t *C, int val)
{
    C->target = val;
    C->count = 0;
}

static inline
int armas_counter_inc(armas_counter_t *C)
{
    int c, t;
    if (C->target == 0 || C->count == C->target)
        return -1;

    if ((c = atomic_inc(&C->count)) == C->target) {
        // reset target to stop counter; 
        t = xchg32(&C->target, 0);
        assert(t == c);
        // wake all threads waiting on target
        futex_wake(&C->target, INT_MAX);
    }
    return 0;
}

static inline
int armas_counter_wait(armas_counter_t *C)
{
    int t = C->target;
    if (t != 0 && t != C->count) {
        futex_wait(&C->target, t, NULL);
        // now target should be zero and count the original target value;
        assert(C->target == 0 && C->count == t);
    }
    return C->count;
}

#endif  /* ARMAS_SYNC_H */
