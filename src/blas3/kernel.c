
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>

#include <armas/armas.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__kernel_inner) && defined(__kernel_colblk_inner) && \
    defined(__kernel_colwise_inner_no_scale) &&   defined(__kernel_colwise_inner_scale_c)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matcpy.h"

#if defined(COMPLEX128)
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */
  #if defined(__AVX__) 
    #include "mult_avx_c128.h"
  #elif defined(__SSE__) 
    #include "mult_sse_c128.h"
  #else
    #include "mult_nosimd.h"
  #endif

#elif defined(COMPLEX64)
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */

  #if defined(__AVX__) //&& defined(WITH_AVX_C64)
    #include "mult_avx_c64.h"
  #elif defined(__SSE__) //&& defined(WITH_SSE_C64)
    #include "mult_sse_c64.h"
  #else
    #include "mult_nosimd.h"
  #endif

#elif defined(FLOAT32)
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */
  #if defined(__x86_64__)
    #if defined(__AVX__) 
      #include "mult_avx_f32.h"
    #elif defined(__SSE__) 
      #include "mult_sse_f32.h"
    #else
      #include "mult_nosimd.h"
    #endif
  #elif defined(__arm__)
    #if defined(__ARM_NEON)
      #if defined(__ARM_FEATURE_FMA)
        #include "mult_armneon_fma_f32.h"
      #else
        #include "mult_armneon_f32.h"
      #endif
    #else
      #include "mult_nosimd.h"
    #endif
  #else
      #include "mult_nosimd.h"
  #endif

#else  
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
  #if defined(__FMA__)
    #include "mult_fma_f64.h"
  #elif defined(__AVX__)
    #include "mult_avx_f64.h"
  #elif defined(__SSE__)
    #include "mult_sse_f64.h"
  #else
    #include "mult_nosimd.h"
  #endif

#endif


/* ---------------------------------------------------------------------------------
 * 
 */

#include "kernel.h"

// update C block defined by nR rows, nJ columns, nP is A, B common dimension
// A, B data arranged for DOT operations, A matrix is the inner matrix block
// and is looped over nJ times
void __kernel_colblk_inner(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                           DTYPE alpha, int nJ, int nR, int nP)
{
  register int j;
  
  for (j = 0; j < nJ-3; j += 4) {
    __CMULT4(Cblk, Ablk, Bblk, alpha, j, nR, nP);
  }
  if (j == nJ)
    return;
  // the uneven column stripping part ....
  if (j < nJ-1) {
    __CMULT2(Cblk, Ablk, Bblk, alpha, j, nR, nP);
    j += 2;
  }
  if (j < nJ) {
    __CMULT1(Cblk, Ablk, Bblk, alpha, j, nR, nP);
    j++;
  }
}

void __kernel_colblk_inner2(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                            DTYPE alpha, int nJ, int nR, int nP, int rb)
{
  register int j, kp, nK;
  mdata_t Ca, Ac;

  for (kp = 0; kp < nR; kp += rb) {
    nK = min(rb, nR-kp);
    __subblock(&Ca, Cblk, kp, 0);
    __subblock(&Ac, Ablk, 0, kp);
    for (j = 0; j < nJ-3; j += 4) {
      __CMULT4(&Ca, &Ac, Bblk, alpha, j, nK, nP);
    }
    if (j == nJ)
      return;
    // the uneven column stripping part ....
    if (j < nJ-1) {
      __CMULT2(&Ca, &Ac, Bblk, alpha, j, nK, nP);
      j += 2;
    }
    if (j < nJ) {
      __CMULT1(&Ca, &Ac, Bblk, alpha, j, nK, nP);
      j++;
    }
  }
}

// update block of C with A and B panels; A panel is nR*P, B panel is P*nSL
// C block is nR*nSL
void __kernel_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                     DTYPE alpha, int flags, int P, int nSL, int nRE,
                                     cache_t *cache)
{
  int j, ip, jp, kp, rp, nP, nI, nJ, nR;
  mdata_t Ca, Ac, Bc;
  Ca.step = C->step;

  if (nRE <= 0 || nSL <= 0)
    return;

  // protect against invalid rb parameter (for time being)
  if (cache->rb < 0 || cache->rb > cache->MB) {
    cache->rb = cache->MB/2;
  }

  for (jp = 0; jp < nSL; jp += cache->NB) {
    // in panels of N columns of C, B
    nJ = min(cache->NB, nSL-jp);

    for (kp = 0; kp < P; kp += cache->KB) {
      nP = min(cache->KB, P-kp);
    
      if (flags & (ARMAS_TRANSB|ARMAS_CONJB)) {
        __subblock(&Bc, B, jp, kp); 
        __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nP, flags);
      } else {
        __subblock(&Bc, B, kp, jp);
        __CPBLK(&cache->Bcpy, &Bc, nP, nJ, flags);
      }
	  
      for (ip = 0; ip < nRE; ip += cache->MB) {
        nI = min(cache->MB, nRE-ip);
        if (flags & (ARMAS_TRANSA|ARMAS_CONJA)) {
          __subblock(&Ac, A, kp, ip);
          __CPBLK(&cache->Acpy, &Ac, nP, nI, flags);
        } else {
          __subblock(&Ac, A, ip, kp);
          __CPBLK_TRANS(&cache->Acpy, &Ac, nI, nP, flags);
        }

        for (rp = 0; rp < nI; rp += cache->rb) {
          nR = min(cache->rb, nI-rp);
          __subblock(&Ca, C, ip+rp, jp);
          __subblock(&Ac, &cache->Acpy, 0, rp);
          for (j = 0; j < nJ-3; j += 4) {
            __CMULT4(&Ca, &Ac, &cache->Bcpy, alpha, j, nR, nP);
          }
          if (j == nJ)
            continue;
          // the uneven column stripping part ....
          if (j < nJ-1) {
            __CMULT2(&Ca, &Ac, &cache->Bcpy, alpha, j, nR, nP);
            j += 2;
          }
          if (j < nJ) {
            __CMULT1(&Ca, &Ac, &cache->Bcpy, alpha, j, nR, nP);
            j++;
          }
        }
      }
    }
  }
}


void __kernel_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                    DTYPE alpha, DTYPE beta, int flags,
                                    int P, int S, int L, int R, int E, cache_t *cache)
{
  int j, ip, jp, kp, rp, nP, nI, nJ, nR;
  mdata_t Ca;
  int KB, NB, MB;
  mdata_t Ac, Bc;

  if (L-S <= 0 || E-R <= 0 || P <= 0)
    return;

  Ca.step = C->step;
  KB = cache->KB; NB = cache->NB; MB = cache->MB;

  // loop over columns of C
  for (jp = S; jp < L; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, L-jp);
	
    // scale C block here.... jp, jp+nJ columns, E-R rows
    __subblock(&Ca, C, R, jp);
    __blk_scale(&Ca, beta, E-R, nJ);

    for (kp = 0; kp < P; kp += KB) {
      nP = min(KB, P-kp);

      if (flags & (ARMAS_TRANSB|ARMAS_CONJB)) {
        __subblock(&Bc, B, jp, kp);
        __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nP, flags);
      } else {
        __subblock(&Bc, B, kp, jp);
        __CPBLK(&cache->Bcpy, &Bc, nP, nJ, flags);
      }
	  
      for (ip = R; ip < E; ip += MB) {
        nI = min(MB, E-ip);
        if (flags & (ARMAS_TRANSA|ARMAS_CONJA)) {
          __subblock(&Ac, A, kp, ip);
          __CPBLK(&cache->Acpy, &Ac, nP, nI, flags);
        } else {
          __subblock(&Ac, A, ip, kp);
          __CPBLK_TRANS(&cache->Acpy, &Ac, nI, nP, flags);
        }

        // workout through A in blocks of cache->rb
        for (rp = 0; rp < nI; rp += cache->rb) {
          nR = min(cache->rb, nI-rp);
          __subblock(&Ca, C, ip+rp, jp);
          __subblock(&Ac, &cache->Acpy, 0, rp);
          for (j = 0; j < nJ-3; j += 4) {
            __CMULT4(&Ca, &Ac, &cache->Bcpy, alpha, j, nR, nP);
          }
          if (j == nJ)
            continue;
          // the uneven column stripping part ....
          if (j < nJ-1) {
            __CMULT2(&Ca, &Ac, &cache->Bcpy, alpha, j, nR, nP);
            j += 2;
          }
          if (j < nJ) {
            __CMULT1(&Ca, &Ac, &cache->Bcpy, alpha, j, nR, nP);
            j++;
          }
        }
      }
    }
  }
}

void __kernel_inner(mdata_t *C, const mdata_t *A, const mdata_t *B,
                     DTYPE alpha, DTYPE beta, int flags,
                     int P, int S, int L, int R, int E, 
                     int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  mdata_t Aa; //, Ba;
  cache_t cache;
  //cache_t cache;

  if (L-S <= 0 || E-R <= 0) {
    // nothing to do, zero columns or rows
    return;
  }

  if (alpha == 0.0) {
    if (beta != 1.0) {
      __subblock(&Aa, C, R, S);
      __blk_scale(&Aa, beta, E-R, L-S);
    }
    return;
  }

  armas_cache_setup2(&cache, cbuf, MB, NB, KB, sizeof(DTYPE));
  // update C using A as inner most matrix
  __kernel_colwise_inner_scale_c(C, A, B, alpha, beta, flags,
                                 P, S, L, R, E, &cache); 
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
