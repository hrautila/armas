

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include <armas/armas.h>
#include "dtype.h"
#include "internal.h"

/*
 * Given a per worker cache buffer size in bytes
 *   - allocate memory block that is multiple of CPU cacheline size and at least
 *     given size when aligned to cacheline.
 *   - compute cache aligned start address
 *   
 *  For standard precision computations divide the cache buffer to two buffers
 *  that hold current source operand blocks. (configuration elements mb, nb, kb)
 *  (mb-x-nb) result is computed from (mb-x-kb) * (kb-x-nb). In order to access
 *  elements in increasing memory order we organise the cache data to computation
 *  (kb-x-mb).T * (kb-x-nb)
 *  So we need to intermediate blocks with row stride of kb. 
 */

#define CACHELINE 64
#define CLSHIFT   5
#define CACHEMIN  1024

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
 * @param cbuf [out]    Cache buffer
 * @param cmem [in]     Requested aligned size
 * @param limem [in]    L1 cache size
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
 * @brief Make cache buffer from provided buffer space
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

/*
 * Divide buffer with size 'S' bytes to 2 blocks of size 'k*m' and 'k*n' items where
 * item size is 'p' bytes.
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
void armas_cbuf_split2(armas_cbuf_t *cbuf,
                       void **aptr, void **bptr, size_t *mb, size_t *nb, size_t *kb, size_t p)
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

/*
 * Setup cache for two-way split buffer
 */
void armas_cache_setup2(cache_t *cache, armas_cbuf_t *cbuf, size_t mb, size_t nb, size_t kb, size_t p)
{
    cache->MB = mb;
    cache->NB = nb;
    cache->KB = kb;

    armas_cbuf_split2(cbuf,
                      (void **)&cache->Acpy.md,
                      (void **)&cache->Bcpy.md,
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
    
    cache->Acpy.step = cache->KB;
    cache->Bcpy.step = cache->KB;
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
void armas_cbuf_split3(armas_cbuf_t *cbuf, void **aptr, void **bptr, void **cptr,
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


void armas_cache_setup3(cache_t *cache, armas_cbuf_t *cbuf, size_t mb, size_t nb, size_t kb, size_t p)
{
    cache->MB = mb;
    cache->NB = nb;
    cache->KB = kb;

    armas_cbuf_split3(cbuf,
                      (void **)&cache->Acpy.md,
                      (void **)&cache->Bcpy.md,
                      (void **)&cache->dC.md,
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
    
    cache->Acpy.step = cache->KB;
    cache->Bcpy.step = cache->KB;
    cache->dC.step   = cache->KB;
    cache->cbuf = cbuf;
}

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
