
#ifndef ARMAS_CACHE_H
#define ARMAS_CACHE_H 1

#include <stdint.h>
#include "armas.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CACHELINE
#define CACHELINE 64
#endif
#ifndef CLSHIFT
#define CLSHIFT   5
#endif
#ifndef CACHEMIN
#define CACHEMIN  1024
#endif
typedef struct cache {
    DTYPE *Acpy;  // kb*mb
    DTYPE *Bcpy;  // kb*nb
    size_t ab_step;
    size_t KB;
    size_t NB;
    size_t MB;
    size_t rb;
    // for extended precision versions
    DTYPE *C0;    // kb*mb; kb >= nb;
    DTYPE *dC;    // kb*mb; kb >= nb
    size_t c_step;
    struct armas_cbuf *cbuf;
} cache_t;

// TODO: here the uint64_t is not right for 32bit platforms

extern inline __attribute__((__always_inline__))
size_t armas_cacheline_adjusted(size_t nbytes)
{
    return ((nbytes + CACHELINE - 1) >> CLSHIFT) << CLSHIFT;
}

extern inline __attribute__((__always_inline__))
size_t armas_cache_alloc_size(size_t requested)
{
    // make it multiple of cacheline size
    size_t asz = ((requested + CACHELINE - 1) >> CLSHIFT) << CLSHIFT;
    // allow for pointer alignment
    return  asz + CACHELINE - 1;
}

extern inline __attribute__((__always_inline__))
size_t armas_cache_aligned(void **aligned, void *ptr, size_t len)
{
    char *p = (char *)ptr;
    uint64_t na, addr = (uint64_t)ptr;
    na = CACHELINE - (addr & (CACHELINE-1));
    *aligned = (void *)&p[na];
    return len - na;
}

extern void cache_setup(cache_t *cache, armas_cbuf_t *cbuf, size_t p, int three);

#ifdef __cplusplus
}
#endif

#endif  // __ARMAS_CACHE_H
