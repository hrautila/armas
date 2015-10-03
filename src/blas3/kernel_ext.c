
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/armas package. It is free software,
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
  #if defined(__AVX__) // && defined(__SIMD_EFT) 
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

  #if defined(__AVX__) // && defined(__SIMD_EFT) 
    #include "mult_avx_f64.h"
    #include "mult_ext_avx_f64.h"
  #else
    #include "mult_nosimd.h"
    #include "mult_ext_nosimd.h"
  #endif   // defined(__AVX__)

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


// update C block defined by nR rows, nJ columns, nP is A, B common dimension
// A, B data arranged for DOT operations, A matrix is the inner matrix block
// and is looped over nJ times
void __kernel_ext_colblk_inner(mdata_t *Cblk, mdata_t *dC,
                               const mdata_t *Ablk, const mdata_t *Bblk,
                               DTYPE alpha, int nJ, int nR, int nP)
{
  register int j;

  for (j = 0; j < nJ-1; j += 2) {
    __CMULT2EXT(Cblk, dC, Ablk, Bblk, alpha, j, nR, nP);
  }
  if (j == nJ)
    return;
  // the uneven column stripping part ....
  __CMULT1EXT(Cblk, dC, Ablk, Bblk, alpha, j, nR, nP);
}

// error free update: C + dC += A*B   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
void __kernel_ext_panel_inner(mdata_t *C, mdata_t *dC,
                              const mdata_t *A, const mdata_t *B,
                              DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache)
{
  int j, kp, nK;
  mdata_t *Acpy, *Bcpy;

  Acpy = cache->Acpy;
  Bcpy = cache->Bcpy;

  for (kp = 0; kp < nP; kp += cache->KB) {
    nK = min(cache->KB, nP-kp);

    if (flags & ARMAS_TRANSB) {
      __CPTRANS(Bcpy->md, Bcpy->step, &B->md[kp*B->step], B->step, nJ, nK);
    } else {
      __CP(Bcpy->md, Bcpy->step, &B->md[kp], B->step, nK, nJ);
    }
	  
    if (flags & ARMAS_TRANSA) {
      __CP(Acpy->md, Acpy->step, &A->md[kp], A->step, nK, nR);
    } else {
      __CPTRANS(Acpy->md, Acpy->step, &A->md[kp*A->step], A->step, nR, nK);
    }

    for (j = 0; j < nJ-1; j += 2) {
      __CMULT2EXT(C, dC, Acpy, Bcpy, alpha, j, nR, nK);
    }
    if (j == nJ)
      continue;
    __CMULT1EXT(C, dC, Acpy, Bcpy, alpha, j, nR, nK);
  }
}

// error free update: C + dC += (A + dA)*B   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
// C + dC += A*B; dC += dA*B
void __kernel_ext_panel_inner_dA(mdata_t *C, mdata_t *dC,
                                 const mdata_t *A, const mdata_t *dA, const mdata_t *B,
                                 DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache)
{
  int j, kp, nK;
  mdata_t *Acpy, *Bcpy;

  Acpy = cache->Acpy;
  Bcpy = cache->Bcpy;

  for (kp = 0; kp < nP; kp += cache->KB) {
    nK = min(cache->KB, nP-kp);

    if (flags & ARMAS_TRANSB) {
      __CPTRANS(Bcpy->md, Bcpy->step, &B->md[kp*B->step], B->step, nJ, nK);
    } else {
      __CP(Bcpy->md, Bcpy->step, &B->md[kp], B->step, nK, nJ);
    }
	  
    if (flags & ARMAS_TRANSA) {
      __CP(Acpy->md, Acpy->step, &A->md[kp], A->step, nK, nR);
    } else {
      __CPTRANS(Acpy->md, Acpy->step, &A->md[kp*A->step], A->step, nR, nK);
    }

    for (j = 0; j < nJ-1; j += 2) {
      // C + dC = A*B
      __CMULT2EXT(C, dC, Acpy, Bcpy, alpha, j, nR, nK);

    }
    if (j == nJ) 
      goto deltaA;

    __CMULT1EXT(C, dC, Acpy, Bcpy, alpha, j, nR, nK);

  deltaA:
    if (flags & ARMAS_TRANSA) {
      __CP(Acpy->md, Acpy->step, &dA->md[kp], dA->step, nK, nR);
    } else {
      __CPTRANS(Acpy->md, Acpy->step, &dA->md[kp*dA->step], dA->step, nR, nK);
    }
    for (j = 0; j < nJ-1; j += 2) {
      // dC += dA*B
      __CMULT2(dC, Acpy, Bcpy, alpha, j, nR, nK);

    }
    if (j == nJ)
      continue;

    // dC += dA*B
    __CMULT1(dC, Acpy, Bcpy, alpha, j, nR, nK);
  }
}


// error free update: C + dC += A*(B + dB)   C is nR,nJ A panel is nR,nP and B panel is nP,nJ
// C + dC += A*B; dC += A*dB
void __kernel_ext_panel_inner_dB(mdata_t *C, mdata_t *dC,
                                 const mdata_t *A, const mdata_t *B, const mdata_t *dB,
                                 DTYPE alpha, int flags, int nJ, int nR, int nP, cache_t *cache)
{
  int j, kp, nK;
  mdata_t *Acpy, *Bcpy;

  Acpy = cache->Acpy;
  Bcpy = cache->Bcpy;

  for (kp = 0; kp < nP; kp += cache->KB) {
    nK = min(cache->KB, nP-kp);

    if (flags & ARMAS_TRANSB) {
      __CPTRANS(Bcpy->md, Bcpy->step, &B->md[kp*B->step], B->step, nJ, nK);
    } else {
      __CP(Bcpy->md, Bcpy->step, &B->md[kp], B->step, nK, nJ);
    }
	  
    if (flags & ARMAS_TRANSA) {
      __CP(Acpy->md, Acpy->step, &A->md[kp], A->step, nK, nR);
    } else {
      __CPTRANS(Acpy->md, Acpy->step, &A->md[kp*A->step], A->step, nR, nK);
    }

    for (j = 0; j < nJ-1; j += 2) {
      // C + dC = A*B
      __CMULT2EXT(C, dC, Acpy, Bcpy, alpha, j, nR, nK);
    }
    if (j == nJ)
      goto deltaB;

    // odd row/col
    __CMULT1EXT(C, dC, Acpy, Bcpy, alpha, j, nR, nK);

  deltaB:
    if (flags & ARMAS_TRANSB) {
      __CPTRANS(Bcpy->md, Bcpy->step, &dB->md[kp*dB->step], dB->step, nJ, nK);
    } else {
      __CP(Bcpy->md, Bcpy->step, &dB->md[kp], dB->step, nK, nJ);
    }

    for (j = 0; j < nJ-1; j += 2) {
      // dC += A*dB
      __CMULT2(dC, Acpy, Bcpy, alpha, j, nR, nK);
    }

    if (j == nJ)
      continue;

    // dC += A*dB
    __CMULT1(dC, Acpy, Bcpy, alpha, j, nR, nK);
  }
}

// update block of C with A and B panels; A panel is nR*P, B panel is P*nSL
// C block is nR*nSL
void __kernel_ext_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                         DTYPE alpha, int flags, int P, int nSL, int nRE,
                                         cache_t *cache)
{
  int j, ip, jp, kp, nP, nI, nJ;
  mdata_t Ca;

  int KB, NB, MB;
  mdata_t *Acpy, *Bcpy, *C0, *dC;

  if (nRE <= 0 || nSL <= 0)
    return;

  Ca.step = C->step;
  KB = cache->KB; NB = cache->NB; MB = cache->MB;
  Acpy = cache->Acpy;
  Bcpy = cache->Bcpy;
  C0   = cache->C0;
  dC   = cache->dC;

  // loop over columns of C
  for (jp = 0; jp < nSL; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, nSL-jp);
	
    for (ip = 0; ip < nRE; ip += MB) {
      nI = min(MB, nRE-ip);

      __subblock(&Ca, C, ip, jp);
      // initialize block {C0, dC} = {C, 0}
      __blk_scale_ext(C0, dC, &Ca, __ZERO, nI, nJ);

      for (kp = 0; kp < P; kp += KB) {
        nP = min(KB, P-kp);

        if (flags & ARMAS_TRANSB) {
          __CPTRANS(Bcpy->md, Bcpy->step, &B->md[jp+kp*B->step], B->step, nJ, nP);
        } else {
          __CP(Bcpy->md, Bcpy->step, &B->md[kp+jp*B->step], B->step, nP, nJ);
        }
	  
        if (flags & ARMAS_TRANSA) {
          __CP(Acpy->md, Acpy->step, &A->md[kp+ip*A->step], A->step, nP, nI);
        } else {
          __CPTRANS(Acpy->md, Acpy->step, &A->md[ip+kp*A->step], A->step, nI, nP);
        }

        for (j = 0; j < nJ-1; j += 2) {
            __CMULT2EXT(C0, dC, Acpy, Bcpy, alpha, j, nI, nP);
        }
        if (j == nJ)
          continue;
        __CMULT1EXT(C0, dC, Acpy, Bcpy, alpha, j, nI, nP);
      }

      // merge back to target
      __blk_merge_ext(&Ca, C0, dC, nI, nJ);
      
    }
  }
}


void __kernel_ext_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                        DTYPE alpha, DTYPE beta, int flags,
                                        int P, int S, int L, int R, int E, cache_t *cache)
{
  int j, ip, jp, kp, nP, nI, nJ;
  mdata_t Ca;
  int KB, NB, MB;
  mdata_t *Acpy, *Bcpy, *C0, *dC;

  if (L-S <= 0 || E-R <= 0 || P <= 0)
    return;

  Ca.step = C->step;
  KB = cache->KB; NB = cache->NB; MB = cache->MB;
  Acpy = cache->Acpy;
  Bcpy = cache->Bcpy;
  C0   = cache->C0;
  dC   = cache->dC;

  // loop over columns of C
  for (jp = S; jp < L; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, L-jp);
	
    for (ip = R; ip < E; ip += MB) {
      nI = min(MB, E-ip);

      __subblock(&Ca, C, ip, jp);
      // compute C0 + dC = beta*C
      __blk_scale_ext(C0, dC, &Ca, beta, nI, nJ);

      for (kp = 0; kp < P; kp += KB) {
        nP = min(KB, P-kp);

        if (flags & ARMAS_TRANSB) {
          __CPTRANS(Bcpy->md, Bcpy->step, &B->md[jp+kp*B->step], B->step, nJ, nP);
        } else {
          __CP(Bcpy->md, Bcpy->step, &B->md[kp+jp*B->step], B->step, nP, nJ);
        }
	  
        if (flags & ARMAS_TRANSA) {
          __CP(Acpy->md, Acpy->step, &A->md[kp+ip*A->step], A->step, nP, nI);
        } else {
          __CPTRANS(Acpy->md, Acpy->step, &A->md[ip+kp*A->step], A->step, nI, nP);
        }

        for (j = 0; j < nJ-1; j += 2) {
            __CMULT2EXT(C0, dC, Acpy, Bcpy, alpha, j, nI, nP);
        }
        if (j == nJ)
          continue;
        __CMULT1EXT(C0, dC, Acpy, Bcpy, alpha, j, nI, nP);
      }

      // merge back to target
      __blk_merge_ext(&Ca, C0, dC, nI, nJ);
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
                       int KB, int NB, int MB)
{
  mdata_t Aa, Ba, Ca, dC;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB/4], Bbuf[MAX_KB*MAX_NB/4] __attribute__((aligned(64)));
  DTYPE Cbuf[MAX_MB*MAX_NB/4], Dbuf[MAX_MB*MAX_NB/4] __attribute__((aligned(64)));

  if (L-S <= 0 || E-R <= 0) {
    // nothing to do, zero columns or rows
    return 0;
  }

  // restrict block sizes as data is copied to aligned buffers of
  // predefined max sizes.
  if (NB > MAX_NB/2 || NB <= 0) {
    NB = MAX_NB/2;
  }
  if (MB > MAX_MB/2 || MB <= 0) {
    MB = MAX_MB/2;
  }
  if (KB  > MAX_KB/2 || KB <= 0) {
    KB = MAX_KB/2;
  }

  if (alpha == 0.0) {
    __subblock(&Aa, C, R, S);
    __blk_scale(&Aa, beta, E-R, L-R);
    return 0;
  }

  // clear Abuf, Bbuf to avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));

  // setup cache area
  Aa = (mdata_t){Abuf, MAX_KB/2};
  Ba = (mdata_t){Bbuf, MAX_KB/2};
  Ca = (mdata_t){Cbuf, MAX_MB/2};
  dC = (mdata_t){Dbuf, MAX_MB/2};
  cache = (cache_t){&Aa, &Ba, KB, NB, MB, &Ca, &dC};

  // update C using A as inner most matrix
  __kernel_ext_colwise_inner_scale_c(C, A, B, alpha, beta, flags,
                                     P, S, L, R, E, &cache); 
  return 0;
}

  
#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
