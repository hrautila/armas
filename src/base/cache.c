
// Copyright (c) Harri Rautila, 2015-2020

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file
 * Cache memory functions.
 */
#include <sys/types.h>
#include <stdint.h>
#include <assert.h>

#include "armas.h"
#include "dtype.h"
#include "cache.h"

/*
 * Given a per worker cache buffer size in bytes
 *   - allocate memory block that is multiple of CPU cacheline size and at least
 *     given size when aligned to cacheline.
 *   - compute cache aligned start address
 *
 * For standard precision computations divide the cache buffer to two buffers
 * that hold current source operand blocks. (configuration elements mb, nb, kb)
 * (mb-x-nb) result is computed from (mb-x-kb) * (kb-x-nb). In order to access
 * elements in increasing memory order we organise the cache data to computation
 * (kb-x-mb).T * (kb-x-nb)
 * So we need to intermediate blocks with row stride of kb.
 */


static inline __attribute__((__always_inline__))
size_t cacheline_adjusted(size_t nbytes)
{
    return ((nbytes + CACHELINE - 1) >> CLSHIFT) << CLSHIFT;
}

static inline __attribute__((__always_inline__))
size_t alloc_size(size_t requested)
{
    // make it multiple of cacheline size
    size_t asz = ((requested + CACHELINE - 1) >> CLSHIFT) << CLSHIFT;
    // allow for pointer alignment
    return  asz + CACHELINE - 1;
}

static inline __attribute__((__always_inline__))
size_t cache_aligned(void **aligned, void *ptr, size_t len)
{
    char *p = (char *)ptr;
    uint64_t na, addr = (uint64_t)ptr;
    na = CACHELINE - (addr & (CACHELINE-1));
    *aligned = (void *)&p[na];
    return len - na;
}

/**
 * @brief Initialize cache buffer
 *
 * Allocates a memory block of at least cmem bytes. Makes cache buffer to start
 * on first cache line aligned element of allocated memory block.
 *
 * @param [out] cbuf   Cache buffer
 * @param [in]  cmem   Requested aligned size
 * @param [in]  l1mem  L1 cache size
 *
 * @returns
 *   Null if initialization fails, otherwise pointer to initialized cache buffer.
 */
armas_cbuf_t *armas_cbuf_init(armas_cbuf_t *cbuf, size_t cmem, size_t l1mem)
{
    size_t asize = alloc_size(cmem);
    if (asize < CACHEMIN)
        asize = CACHEMIN;
    cbuf->__unaligned = calloc(asize, 1);
    if (! cbuf->__unaligned)
        return (armas_cbuf_t *)0;
    cbuf->__nbytes = asize;
    cbuf->len = cache_aligned((void **)&cbuf->data, cbuf->__unaligned, asize);
    cbuf->cmem = cmem;
    cbuf->l1mem = l1mem;
    return cbuf;
}

/**
 * @brief Create new cache buffer instance.
 */
armas_cbuf_t *armas_sbuf_new(size_t cmem, size_t l1mem)
{
    armas_cbuf_t *cb = calloc(1, sizeof(struct armas_cbuf));
    if (!cb)
        return cb;
    if (!armas_cbuf_init(cb, cmem, l1mem)) {
        free(cb);
        cb = (struct armas_cbuf *)0;
    }
    return cb;
}

/**
 * @brief Make cache buffer from provided buffer space
 *
 * Makes cache buffer to start on first cache line aligned element of
 * provided memory block.
 *
 * @param [out] cbuf   Cache buffer
 * @param [in]  buf    Buffer space, at least cmem+CACHELINE bytes
 * @param [in]  cmem   Requested aligned size
 * @param [in]  l1mem  L1 cache size
 *
 * @returns
 *      Initialized cache buffer.
 */
armas_cbuf_t *armas_cbuf_make(armas_cbuf_t *cbuf, void *buf, size_t cmem, size_t l1mem)
{
    cbuf->__unaligned = (void *)0;
    cbuf->__nbytes = 0;
    cbuf->len = cache_aligned((void **)&cbuf->data, buf,  cmem);
    cbuf->cmem = cmem;
    cbuf->l1mem = l1mem;
    return cbuf;
}

/**
 * @brief Release buffer resources.
 */
void armas_cbuf_release(armas_cbuf_t *cbuf)
{
    if (cbuf && cbuf->__unaligned) {
        free(cbuf->__unaligned);
        cbuf->__unaligned = (void *)0;
        cbuf->data = (void *)0;
    }
}

/**
 * @brief Free cache buffer and its resources.
 */
void armas_cbuf_free(armas_cbuf_t *cbuf)
{
    if (cbuf) {
        armas_cbuf_release(cbuf);
        free(cbuf);
    }
}

/*
 * Divide buffer with size 'S' bytes to 2 blocks of size 'k*m' and 'k*n' items
 * where item size is 'p' bytes.
 *
 *  (m + n)*k = S/p and m/n = a, k/n = b => n = k/b
 *
 *  (m/n + 1)*n*k = S/p =>
 *  n*k = (S/p)/(1+a)   =>
 *  k^2 = b*(S/p)/(1+a) =>
 *  k   = sqrt(b*(S/p)/(1+a))
 *
 *  ensure that
 *   1. k*p mod CLsize == 0
 *   2. n mod 4 == 0
 *   3  m mod 4 == 0
 *
 *  assume: kb >= nb; nb >= mb
 */
/**
 * @brief Two way split of cache buffer.
 *
 * Divide cache buffer two blocks of size 'kb*mb' and 'kb*nb' items where
 * item size is 'p' bytes. Ensure that each columns of kb items starts
 * at cpu cache line.(ie. kb*p mod CACHELINE == 0).
 *
 * @param [in] cbuf
 *      Cache buffer
 * @param [out] aptr
 *      Pointer to first cache block. If null no value is returned.
 * @param [out] bptr
 *      Pointer to second cache block. If null no value is returned.
 * @param [in,out] mb
 *      On entry initial size of mb, on exit calculated cache line aligned size
 * @param [in,out] nb
 *      On entry initial size of nb, on exit calculated cache line aligned size
 * @param [in,out] kb
 *      On entry initial size of kb, on exit calculated cache line aligned size
 * @param [in] p
 *      item size in bytes
 */
void armas_cbuf_split2(armas_cbuf_t *cbuf,
                       void **aptr, void **bptr, size_t *mb,
                       size_t *nb, size_t *kb, size_t p)
{
    size_t clp, kn, mn, nn, S = cbuf->len/p;
    double a, b;

    if (cbuf->len <= CACHEMIN) {
        mn = nn = kn = p < sizeof(double) ? 8 : 4;
    } else {
        a = (double)(*mb)/(*nb);
        b = (double)(*kb)/(*nb);
        kn = (size_t)sqrt(b*S/(1.0+a));

        clp = CACHELINE/p;
        if (kn > clp)
            kn -= kn % clp;

        nn = (size_t)((double)kn/b);
        mn = (size_t)((double)nn*a);
        // make multiples of 4; mn and nn are 'column' counts, kn is 'row' count
        nn &= ~0x3;
        mn &= ~0x3;
    }
    *mb = mn;
    *nb = nn;
    *kb = kn;

    if (aptr)
        *aptr = (void *)cbuf->data;
    if (bptr)
        *bptr = (void *)&cbuf->data[mn*kn*p];
    // *aptr and *bptr aligned to cacheline
}

static
int armas_cache_fixed2(cache_t *cache, armas_cbuf_t *cbuf,
                       size_t mb, size_t nb, size_t kb, size_t p)
{
    size_t bytes = (kb > (CACHELINE/p) ? kb : (CACHELINE/p))*(mb + nb)*p;
    if (bytes > cbuf->cmem) {
        return -1;
    }
    if (kb > (CACHELINE/p))
        kb -= kb % (CACHELINE/p);
    if (kb < 4)
        kb = 4;
    mb &= ~0x3;
    nb &= ~0x3;
    cache->KB = kb;
    cache->MB = mb;
    cache->NB = nb;
    cache->rb = mb/4;
    if (cache->rb < 4)
        cache->rb = 4;
    if (kb > (CACHELINE/p))
        cache->ab_step = cache->c_step = kb;
    else
        cache->ab_step = cache->c_step = (CACHELINE/p);

    cache->Acpy = (DTYPE *)cbuf->data;
    cache->Bcpy = (DTYPE *)(&cbuf->data[mb*cache->ab_step*p]);
    cache->dC = (DTYPE *)0;
    cache->ab_step = cache->c_step = kb > (CACHELINE/p) ? kb : (CACHELINE/p);
    cache->cbuf = cbuf;
    return 0;
}

/**
 * \brief Setup cache for two-way split cache buffer
 *
 */
void armas_cache_setup2(cache_t *cache, armas_cbuf_t *cbuf,
                        size_t mb, size_t nb, size_t kb, size_t p)
{
    cache->MB = mb;
    cache->NB = nb;
    cache->KB = kb;

    armas_cbuf_split2(cbuf,
                      (void **)&cache->Acpy,
                      (void **)&cache->Bcpy,
                      &cache->MB, &cache->NB, &cache->KB, p);

    // adjust row count for inner most loop
    if (cbuf->l1mem > 0) {
        cache->rb = cbuf->l1mem / (cache->KB*p);
        cache->rb &= ~0x3;
        if (cache->rb == 0)
            cache->rb = 4;
        if (cache->rb > cache->MB)
            cache->rb = cache->MB;
    } else {
        cache->rb = cache->MB;
    }

    if (cache->KB < (CACHELINE/p))
        cache->ab_step =  CACHELINE/p;
    else
        cache->ab_step = cache->KB;
    cache->c_step = 0;
    cache->cbuf = cbuf;
}

/*
 * Three-way split of cache buffer
 *
 *  +--m----+---n---+---n---+
 *  |       |       |       |    A  = k*m
 *  |  A    |  B    |  dC   |    B  = k*n
 *  |       |       |       |    dC = m*n  ==> requires m <= k
 *  +-------+-------+-------+
 *
 *  k*(m + 2n) = B , m/n = a,  k/n = b <=> n = k/b,  m <= k ==> a <= b
 *  k*n*(m/n + 2) == (k^2/b)*(a + 2) = S
 *  k = sqrt(b*S/(2 + a))
 */
void armas_cbuf_split3(armas_cbuf_t *cbuf,
                       void **aptr, void **bptr, void **cptr,
                       size_t *mb, size_t *nb, size_t *kb, size_t p)
{
    size_t clp, kn, mn, nn, S = cbuf->len/p;
    double a, b;

    if (cbuf->len <= CACHEMIN) {
        mn = nn = kn = p < sizeof(double) ? 8 : 4;
    } else {
        a = (double)(*mb)/(*nb);
        b = (double)(*kb)/(*nb);
        // require mn <= kn ==> a <= b
        if (b < a)
            b = a;

        kn = (size_t)sqrt(b*S/(2.0+a));

        clp = CACHELINE/p;
        if (kn > clp)
            kn -= kn % clp;

        nn = (size_t)((double)kn/b);
        mn = (size_t)((double)nn*a);
        // make multiples of 4; mn and nn are 'column' counts, kn is 'row' count
        nn &= ~0x3;
        mn &= ~0x3;
    }
    *mb = mn;
    *nb = nn;
    *kb = kn;

    if (aptr)
        *aptr = (void *)cbuf->data;
    if (bptr)
        *bptr = (void *)&cbuf->data[mn*kn*p];
    if (cptr)
        *cptr = (void *)&cbuf->data[(mn+nn)*kn*p];
}

/*
 * Setup three-way split up cache buffer with fixed block sizes.
 * If provided memory block is too small return with error.
 */
static
int armas_cache_fixed3(cache_t *cache, armas_cbuf_t *cbuf,
                       size_t mb, size_t nb, size_t kb, size_t p)
{
    size_t bytes = (kb > (CACHELINE/p) ? kb : (CACHELINE/p))*(mb + 2*nb)*p;
    if (bytes > cbuf->cmem) {
        return -1;
    }
    if (kb > (CACHELINE/p))
        kb -= kb % (CACHELINE/p);
    if (kb < 4)
        kb = 4;
    mb &= ~0x3;
    nb &= ~0x3;
    cache->KB = kb;
    cache->MB = mb;
    cache->NB = nb;
    cache->rb = mb/4;
    if (cache->rb < 4)
        cache->rb = 4;
    if (kb > (CACHELINE/p))
        cache->ab_step = cache->c_step = kb;
    else
        cache->ab_step = cache->c_step = (CACHELINE/p);

    cache->Acpy = (DTYPE *)cbuf->data;
    cache->Bcpy = (DTYPE *)(&cbuf->data[mb*cache->ab_step*p]);
    cache->dC   = (DTYPE *)(&cbuf->data[(mb+nb)*cache->ab_step*p]);
    cache->cbuf = cbuf;
    return 0;
}

void armas_cache_setup3(cache_t *cache, armas_cbuf_t *cbuf,
                        size_t mb, size_t nb, size_t kb, size_t p)
{
    cache->MB = mb;
    cache->NB = nb;
    cache->KB = kb;

    armas_cbuf_split3(cbuf,
                      (void **)&cache->Acpy,
                      (void **)&cache->Bcpy,
                      (void **)&cache->dC,
                      &cache->MB, &cache->NB, &cache->KB, p);

    // adjust row count for inner most loop
    if (cbuf->l1mem > 0) {
        cache->rb = cbuf->l1mem / (cache->KB*p);
        cache->rb &= ~0x3;
        if (cache->rb == 0)
            cache->rb = 4;
        if (cache->rb > cache->MB)
            cache->rb = cache->MB;
    } else {
        cache->rb = cache->MB;
    }

    if (cache->KB < (CACHELINE/p)) {
        cache->ab_step = CACHELINE/p;
        cache->c_step   = CACHELINE/p;
    } else {
        cache->ab_step = cache->KB;
        cache->c_step   = cache->KB;
    }
    cache->cbuf = cbuf;
}

void armas_cache_setup(cache_t *cache, armas_cbuf_t *cbuf, int three, size_t p)
{
    armas_env_t *env = armas_getenv();
    if (three == 3) {
        if (env->fixed &&
            armas_cache_fixed3(cache, cbuf, env->mb, env->nb, env->kb, p) == 0)
            return;
        armas_cache_setup3(cache, cbuf, env->mb, env->nb, env->kb, p);
    } else {
        if (env->fixed &&
            armas_cache_fixed2(cache, cbuf, env->mb, env->nb, env->kb, p) == 0)
            return;
        armas_cache_setup2(cache, cbuf, env->mb, env->nb, env->kb, p);
    }
}

static __thread void *cbuf_ptr = (void *)0;

extern void armas_on_exit(void *);

/**
 * @brief Create thread global cache memory buffer if not exist.
 */
armas_cbuf_t *armas_cbuf_create_thread_global()
{
    if (cbuf_ptr)
        return (armas_cbuf_t *)cbuf_ptr;

    armas_env_t *env = armas_getenv();
    size_t nbyt = sizeof(armas_cbuf_t) + alloc_size(env->cmem);
    cbuf_ptr = calloc(nbyt, 1);
    if (!cbuf_ptr)
        return (armas_cbuf_t *)0;

    armas_cbuf_t *cb = (armas_cbuf_t *)cbuf_ptr;
    unsigned char *cp = (unsigned char *)cbuf_ptr;
    armas_cbuf_make(cb, &cp[sizeof(armas_cbuf_t)], env->cmem, env->l1mem);

    return cb;
}

/**
 * @brief Get thread global cache memory buffer.
 */
armas_cbuf_t *armas_cbuf_get_thread_global()
{
    return armas_cbuf_create_thread_global();
}

/**
 * @brief Release thread global cache memory buffer.
 */
void armas_cbuf_release_thread_global()
{
    if (cbuf_ptr) {
        void *p = cbuf_ptr;
        cbuf_ptr = 0;
        free(p);
    }
}

/**
 * @brief Select internal cache memory buffer based on configuration.
 */
int armas_cbuf_select(armas_cbuf_t *cbuf, armas_conf_t *cf)
{
    armas_cbuf_t *cb;
    armas_env_t *env = armas_getenv();
    if (cf && cf->work) {
        size_t ws = armas_wbytes(cf->work);
        if (ws >= env->cmem) {
            // use provided cbuf if it is large enough.
            armas_cbuf_make(cbuf, armas_wptr(cf->work), ws, env->l1mem);
            armas_wreserve(cf->work, ws, 1);
            cbuf->__unaligned = (void *)0;
            return 0;
        }
    }
    if (cf && (cf->optflags & ARMAS_CBUF_LOCAL) != 0) {
        return armas_cbuf_init(cbuf, env->cmem, env->l1mem) ? 0 : -1;
    }

    cb = armas_cbuf_create_thread_global();
    if (cb) {
        *cbuf = *cb;
        return 0;
    }
    return -1;
}
