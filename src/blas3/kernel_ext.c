
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <armas/armas.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(__kernel_inner_ext) && \
    defined(__kernel_ext_colwise_inner_no_scale) && defined(__kernel_ext_colwise_inner_scale_c)
#define __ARMAS_PROVIDES 1
#endif

// this this requires no external public functions
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matcpy.h"

#ifdef COMPLEX128
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */
  #include "mult_ext_nosimd.h"
  #include "mult_nosimd.h"

#elif COMPLEX64
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */
  #include "mult_ext_nosimd.h"
  #include "mult_nosimd.h"

#elif FLOAT32
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */
  #if defined(__AVX__) 
    #include "mult_avx_f32.h"
    #include "mult_ext_avx_f32.h"
  #else
    #include "mult_nosimd.h"
    #include "mult_ext_nosimd.h"
  #endif
#else  
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
  #if defined(__AVX__)
    #include "mult_avx_f64.h"
    #include "mult_ext_avx_f64.h"
  #else
    #include "mult_nosimd.h"
    #include "mult_ext_nosimd.h"
  #endif   
#endif

#include "kernel_ext.h"
#include "kernel.h"

// scale block with constant (C + dC = beta*A)
void __blk_scale_ext(mdata_t *C0, mdata_t *dC, const mdata_t *A, DTYPE beta, int nR, int nC)
{
  int i, j;
  if (beta == __ZERO) {
    for (j = 0; j < nC; j++) {
      for (i = 0; i < nR; i++) {
          C0->md[i+j*C0->step] = __ZERO;
          dC->md[i+j*dC->step] = __ZERO;
      }
    }
    return;
  }
  if (beta == __ONE) {
    for (j = 0; j < nC; j++) {
      for (i = 0; i < nR; i++) {
        C0->md[i+j*C0->step] = A->md[i+j*A->step];
        dC->md[i+j*dC->step] = __ZERO;
      }
    }
    return;
  }
  for (j = 0; j < nC; j++) {
    for (i = 0; i < nR-1; i += 2) {
      twoprod(&C0->md[(i+0)+j*C0->step],
              &dC->md[(i+0)+j*dC->step], beta, A->md[(i+0)+j*A->step]);
      twoprod(&C0->md[(i+1)+j*C0->step],
              &dC->md[(i+1)+j*dC->step], beta, A->md[(i+1)+j*A->step]);
    }
    if (i != nR) {
      twoprod(&C0->md[(i+0)+j*C0->step],
              &dC->md[(i+0)+j*dC->step], beta, A->md[(i+0)+j*A->step]);
    }
  }
}

static inline
void __ext_colblk_inner(mdata_t *Cblk, mdata_t *dC,
                          const mdata_t *Ablk, const mdata_t *Bblk,
                          DTYPE alpha, int nJ, int nR, int nP, int rb)
{
  register int j, kp, nK;
  mdata_t Ca, Da, Ac;

  for (kp = 0; kp < nR; kp += rb) {
    nK = min(rb, nR-kp);
    __subblock(&Ca, Cblk, kp, 0);
    __subblock(&Da, dC,   kp, 0);
    __subblock(&Ac, Ablk, 0,  kp);

    for (j = 0; j < nJ-1; j += 2) {
      __CMULT2EXT(&Ca, &Da, &Ac, Bblk, alpha, j, nK, nP);
    }
    if (j == nJ)
      continue;
    // the uneven column stripping part ....
    __CMULT1EXT(&Ca, &Da, &Ac, Bblk, alpha, j, nK, nP);
  }
}

// update C block defined by nR rows, nJ columns, nP is A, B common dimension
// A, B data arranged for DOT operations, A matrix is the inner matrix block
// and is looped over nJ times
void __kernel_ext_colblk_inner(mdata_t *Cblk, mdata_t *dC,
                               const mdata_t *Ablk, const mdata_t *Bblk,
                               DTYPE alpha, int nJ, int nR, int nP, int rb)
{
  __ext_colblk_inner(Cblk, dC, Ablk, Bblk, alpha, nJ, nR, nP, rb);
#if 0
  register int j;

  for (j = 0; j < nJ-1; j += 2) {
    __CMULT2EXT(Cblk, dC, Ablk, Bblk, alpha, j, nR, nP);
  }
  if (j == nJ)
    return;
  // the uneven column stripping part ....
  __CMULT1EXT(Cblk, dC, Ablk, Bblk, alpha, j, nR, nP);
#endif
}

// error free update: C + dC += A*B   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
void __kernel_ext_panel_inner(mdata_t *C, mdata_t *dC,
                              const mdata_t *A, const mdata_t *B,
                              DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache)
{
  int kp, nK;
  mdata_t Bc, Ac;

  for (kp = 0; kp < nP; kp += cache->KB) {
    nK = min(cache->KB, nP-kp);

    if (flags & ARMAS_TRANSB) {
      __subblock(&Bc, B, 0, kp);
      __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nK, flags);
    } else {
      __subblock(&Bc, B, kp, 0);
      __CPBLK(&cache->Bcpy, &Bc, nK, nJ, flags);
    }
	  
    if (flags & ARMAS_TRANSA) {
      __subblock(&Ac, A, kp, 0);
      __CPBLK(&cache->Acpy, &Ac, nK, nR, flags);
    } else {
      __subblock(&Ac, A, 0, kp);
      __CPBLK_TRANS(&cache->Acpy, &Ac, nR, nK, flags);
    }

    __ext_colblk_inner(C, dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nR, nK, cache->rb);
  }
}

// error free update: C + dC += (A + dA)*B   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
// C + dC += A*B; dC += dA*B
void __kernel_ext_panel_inner_dA(mdata_t *C, mdata_t *dC,
                                 const mdata_t *A, const mdata_t *dA, const mdata_t *B,
                                 DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache)
{
  int kp, nK; //, j
  mdata_t /**Acpy, *Bcpy,*/ Bc, Ac;

  for (kp = 0; kp < nP; kp += cache->KB) {
    nK = min(cache->KB, nP-kp);

    if (flags & ARMAS_TRANSB) {
      __subblock(&Bc, B, 0, kp);
      __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nK, flags);
    } else {
      __subblock(&Bc, B, kp, 0);
      __CPBLK(&cache->Bcpy, &Bc, nK, nJ, flags);
    }
	  
    if (flags & ARMAS_TRANSA) {
      __subblock(&Ac, A, kp, 0);
      __CPBLK(&cache->Acpy, &Ac, nK, nR, flags);
    } else {
      __subblock(&Ac, A, 0, kp);
      __CPBLK_TRANS(&cache->Acpy, &Ac, nR, nK, flags);
    }

    __ext_colblk_inner(C, dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nR, nK, cache->rb);

    // update dC += dA*B
    if (flags & ARMAS_TRANSA) {
      __subblock(&Ac, dA, kp, 0);
      __CPBLK(&cache->Acpy, &Ac, nK, nR, flags);
    } else {
      __subblock(&Ac, A, 0, kp);
      __CPBLK_TRANS(&cache->Acpy, &Ac, nR, nK, flags);
    }

    __kernel_colblk_inner(dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nR, nK, cache->rb);
  }
}


// error free update: C + dC += A*(B + dB)   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
// C + dC += A*B; dC += A*dB
void __kernel_ext_panel_inner_dB(mdata_t *C, mdata_t *dC,
                                 const mdata_t *A, const mdata_t *B, const mdata_t *dB,
                                 DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache)
{
  int kp, nK; //, j
  mdata_t /**Acpy, *Bcpy,*/ Ac, Bc;

  for (kp = 0; kp < nP; kp += cache->KB) {
    nK = min(cache->KB, nP-kp);

    if (flags & ARMAS_TRANSB) {
      __subblock(&Bc, B, 0, kp);
      __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nK, flags);
    } else {
      __subblock(&Bc, B, kp, 0);
      __CPBLK(&cache->Bcpy, &Bc, nK, nJ, flags);
    }
	  
    if (flags & ARMAS_TRANSA) {
      __subblock(&Ac, A, kp, 0);
      __CPBLK(&cache->Acpy, &Ac, nK, nR, flags);
    } else {
      __subblock(&Ac, A, 0, kp);
      __CPBLK_TRANS(&cache->Acpy, &Ac, nR, nK, flags);
    }

    __ext_colblk_inner(C, dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nR, nK, cache->rb);

    // dC += A*dB
    if (flags & ARMAS_TRANSB) {
      __subblock(&Bc, dB, 0, kp);
      __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nK, flags);
    } else {
      __subblock(&Bc, dB, kp, 0);
      __CPBLK(&cache->Bcpy, &Bc, nK, nJ, flags);
    }

    __kernel_colblk_inner(dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nR, nK, cache->rb);
  }
}

// update block of C with A and B panels; A panel is nR*P, B panel is P*nSL
// C block is nR*nSL
void __kernel_ext_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                         DTYPE alpha, int flags, int P, int nSL, int nRE,
                                         cache_t *cache)
{
  int ip, jp, kp, nP, nI, nJ; //, j
  mdata_t Ca;

  int KB, NB, MB;
  mdata_t Ac, Bc;

  if (nRE <= 0 || nSL <= 0)
    return;

  Ca.step = C->step;
  KB = cache->KB; NB = cache->NB; MB = cache->MB;

  // loop over columns of C
  for (jp = 0; jp < nSL; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, nSL-jp);
	
    for (ip = 0; ip < nRE; ip += MB) {
      nI = min(MB, nRE-ip);

      __subblock(&Ca, C, ip, jp);
      // initialize block {C0, dC} = {C, 0}
      __blk_scale_ext(&Ca, &cache->dC, &Ca, __ZERO, nI, nJ);

      for (kp = 0; kp < P; kp += KB) {
        nP = min(KB, P-kp);

        if (flags & ARMAS_TRANSB) {
          __subblock(&Bc, B, jp, kp);
          __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nP, flags);

        } else {
          __subblock(&Bc, B, kp, jp);
          __CPBLK(&cache->Bcpy, &Bc, nP, nJ, flags);
        }
	  
        if (flags & ARMAS_TRANSA) {
          __subblock(&Ac, A, kp, ip);
          __CPBLK(&cache->Acpy, &Ac, nP, nI, flags);
        } else {
          __subblock(&Ac, A, ip, kp);
          __CPBLK_TRANS(&cache->Acpy, &Ac, nI, nP, flags);
        }
        __ext_colblk_inner(C, &cache->dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nI, nP, cache->rb);
      }

      // merge back to target
      __blk_merge_ext(&Ca, &Ca, &cache->dC, nI, nJ);
      
    }
  }
}


void __kernel_ext_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                        DTYPE alpha, DTYPE beta, int flags,
                                        int P, int S, int L, int R, int E, cache_t *cache)
{
  int ip, jp, kp, nP, nI, nJ; //, j
  mdata_t Ca;
  int KB, NB, MB;
  mdata_t Bc, Ac;

  if (L-S <= 0 || E-R <= 0 || P <= 0)
    return;

  Ca.step = C->step;
  KB = cache->KB; NB = cache->NB; MB = cache->MB;

  // loop over columns of C
  for (jp = S; jp < L; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, L-jp);
	
    for (ip = R; ip < E; ip += MB) {
      nI = min(MB, E-ip);

      __subblock(&Ca, C, ip, jp);
      // compute C0 + dC = beta*C
      __blk_scale_ext(&Ca, &cache->dC, &Ca, beta, nI, nJ);

      for (kp = 0; kp < P; kp += KB) {
        nP = min(KB, P-kp);

        if (flags & ARMAS_TRANSB) {
          __subblock(&Bc, B, jp, kp);
          __CPBLK_TRANS(&cache->Bcpy, &Bc, nJ, nP, flags);
        } else {
          __subblock(&Bc, B, kp, jp);
          __CPBLK(&cache->Bcpy, &Bc, nP, nJ, flags);
        }
	  
        if (flags & ARMAS_TRANSA) {
          __subblock(&Ac, A, kp, ip);
          __CPBLK(&cache->Acpy, &Ac, nP, nI, flags);
        } else {
          __subblock(&Ac, A, ip, kp);
          __CPBLK_TRANS(&cache->Acpy, &Ac, nI, nP, flags);
        }

        __ext_colblk_inner(C, &cache->dC, &cache->Acpy, &cache->Bcpy, alpha, nJ, nI, nP, cache->rb);
      }
      // merge back to target
      __blk_merge_ext(&Ca, &Ca, &cache->dC, nI, nJ);
    }
  }
}

/*
 * We need 4 intermediate areas: C0, dC where C = C0 + dC, A0 holding current block in A,
 *
 * strategy: [t,p] tiles of C compute for each tile
 *   C = C0 + dC = sum alpha*A[i,k]*B[k,j]
 *      where A[i,:] is row panel of A 
 *      and   B[:,j] is column panel of B
 */
int __kernel_inner_ext(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       DTYPE alpha, DTYPE beta, int flags,
                       int P, int S, int L, int R, int E, 
                       int KB, int NB, int MB, armas_cbuf_t *cbuf)
{
  mdata_t Aa;
  cache_t mcache;

  if (L-S <= 0 || E-R <= 0) {
    // nothing to do, zero columns or rows
    return 0;
  }

  if (alpha == 0.0) {
    if (beta != 1.0) {
      __subblock(&Aa, C, R, S);
      __blk_scale(&Aa, beta, E-R, L-S);
    }
    return 0;
  }

  armas_cache_setup3(&mcache, cbuf, MB, NB, KB, sizeof(DTYPE));
  // update C using A as inner most matrix
  __kernel_ext_colwise_inner_scale_c(C, A, B, alpha, beta, flags,
                                     P, S, L, R, E, &mcache); 
  return 0;
}

  
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
