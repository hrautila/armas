
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_CHAN_H
#define __ARMAS_CHAN_H 1

#include "sync.h"

#define ChannelType(ElemType, Name)		\
  struct Name {					\
    ElemType *elems;				\
    unsigned int size;				\
    unsigned int count;				\
    unsigned int head;				\
    unsigned int tail;				\
    mutex_t mlock;				\
  };						\
  typedef struct Name Name ## _t;

#define ChannelInit(ElemType, Name)			\
  static inline                                         \
  void Name ## _init(struct Name *ch, unsigned int sz)	\
  {							\
    ch->elems = malloc(sz*sizeof(ElemType));		\
    ch->size = sz;					\
    ch->count = 0;					\
    ch->head = 0;					\
    ch->tail = 0;					\
    mutex_init(&ch->mlock, NULL);			\
  }

#define ChannelRelease(Name)			\
  static inline                                 \
  void Name ## _release(struct Name *ch)	\
  {						\
    if (ch && ch->elems)			\
      free(ch->elems);				\
  } 

#define ChannelCreate(Name)				\
  static inline                                         \
  struct Name *Name ## _create(unsigned int sz)		\
  {							\
    struct Name *ch = malloc(sizeof(struct Name));	\
    if (ch)						\
      Name ## _init(ch, sz);				\
    return ch;						\
  }

#define ChannelFree(Name)			\
  static inline                                 \
  void Name ## _free(struct Name *ch)		\
  {						\
    if (ch) {					\
      Name ## _release(ch);                     \
      free(ch);					\
    }						\
  }

#define ChannelEnqueue(ElemType, Name)                  \
  static inline                                         \
  int Name ## _enqueue(struct Name *ch, ElemType v)	\
  {                                                     \
    mutex_lock(&ch->mlock);                             \
    if (ch->count == ch->size) {                        \
      mutex_unlock(&ch->mlock);                         \
      return -1;                                        \
    }                                                   \
    int signal = ch->count == 0;                        \
    ch->elems[ch->tail] = v;                            \
    ch->count++;                                        \
    ch->tail = (ch->tail + 1) % ch->size;               \
    if (signal)                                         \
      futex_wake(&ch->head, 1);                         \
    mutex_unlock(&ch->mlock);                           \
    return 0;                                           \
  }

#define ChannelDequeue(ElemType, Name)			\
  static inline                                         \
  int Name ## _dequeue(struct Name *ch, ElemType *v)	\
  {							\
    mutex_lock(&ch->mlock);				\
    if (ch->count == 0) {				\
      mutex_unlock(&ch->mlock);				\
      return -1;					\
    }							\
    *v = ch->elems[ch->head];				\
    int signal = ch->count == ch->size;			\
    ch->count--;					\
    ch->head = (ch->head + 1) % ch->size;		\
    if (signal)						\
      futex_wake(&ch->tail, 1);				\
    mutex_unlock(&ch->mlock);				\
    return 0;						\
  }

// empty: head == tail && count == 0		
//   before sleeping try to wake up some writer
#define ChannelOnEmpty(Name)                                            \
  static inline                                                         \
  void Name ## _wait_onempty(struct Name *ch, unsigned int usec)	\
  {                                                                     \
    int i;                                                              \
    volatile unsigned int *cntr = &ch->count;                           \
    struct timespec *tp, tm = {0, usec*1000};                           \
    futex_wake(&ch->tail, 1);                                           \
    for (i = 0; i < 100; i++) {                                         \
      if (*cntr != 0)                                                   \
        return;                                                         \
      cpu_relax();                                                      \
    }                                                                   \
    tp = usec == 0 ? NULL : &tm;                                        \
    futex_wait(&ch->head, ch->tail, tp);                                \
  }
  
// full: tail == head && count == size
//   before sleeping try to wake up some reader
#define ChannelOnFull(Name)                                     \
  static inline                                                 \
  void Name ## _wait_onfull(struct Name *ch, unsigned int usec)	\
  {                                                             \
    int i;                                                      \
    volatile unsigned int *cntr = &ch->count;                   \
    struct timespec *tp, tm = {0, 100*1000};                    \
    futex_wake(&ch->head, 1);                                   \
    for (i = 0; i < 100; i++) {                                 \
      if (*cntr != ch->size)                                    \
        return;                                                 \
      cpu_relax();                                              \
    }                                                           \
    tp = usec == 0 ? NULL : &tm;                                \
    futex_wait(&ch->tail, ch->head, tp);                        \
  }

#define ChannelRead(ElemType, Name)			\
  static inline                                         \
  void Name ## _read(struct Name *ch, ElemType *v)	\
  {							\
    while (Name ## _dequeue(ch, v) != 0) {		\
      Name ## _wait_onempty(ch, 100);			\
    }							\
  }

#define ChannelWrite(ElemType, Name)                    \
  static inline                                         \
  void Name ## _write(struct Name *ch, ElemType v)	\
  {							\
    while (Name ## _enqueue(ch, v) != 0) {		\
      Name ## _wait_onfull(ch, 100);                    \
    }							\
  }


#define DefineChannelType(ElemType, Name) \
  ChannelType(ElemType, Name)		  \
  ChannelInit(ElemType, Name)		  \
  ChannelRelease(Name)			  \
  ChannelCreate(Name)			  \
  ChannelFree(Name)			  \
  ChannelEnqueue(ElemType, Name)	  \
  ChannelDequeue(ElemType, Name)	  \
  ChannelOnEmpty(Name)			  \
  ChannelOnFull(Name)			  \
  ChannelRead(ElemType, Name)		  \
  ChannelWrite(ElemType, Name)

#endif  // __ARMAS_CHAN_H

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

