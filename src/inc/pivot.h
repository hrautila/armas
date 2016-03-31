
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.



#ifndef __ARMAS_PIVOT_H
#define __ARMAS_PIVOT_H 1

#include <armas/armas.h>

#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE extern inline
#endif

__ARMAS_INLINE
void __subpivot(armas_pivot_t *pA, armas_pivot_t *pB, int K, int N)
{
  pA->npivots = N;
  pA->indexes = &pB->indexes[K];
  pA->owner = 0;
}
 
/*
 * Partition p to 2 by 1 blocks.
 *
 *        pT
 *  p --> --
 *        pB
 *
 * Parameter nb is initial block size for pT (pTOP) or pB (pBOTTOM).  
 */
__ARMAS_INLINE
void __pivot_2x1(armas_pivot_t *pT, armas_pivot_t *pB,
		 armas_pivot_t *P, int nb, int direction)
{
  switch (direction) {
  case ARMAS_PTOP:
    if (nb > P->npivots)
      nb = P->npivots;
    __subpivot(pT, P, 0, nb);
    __subpivot(pB, P, nb, P->npivots-nb);
    break;
  case ARMAS_PBOTTOM:
    __subpivot(pT, P, 0,  P->npivots-nb);
    __subpivot(pB, P, P->npivots-nb, nb);
    break;
  }
}


/*
 * Repartition 2 by 1 block to 3 by 1 block.
 * 
 *           pT      p0            pT       p0
 * pBOTTOM: --  --> --   ; pTOP:   --  -->  p1
 *           pB      p1            pB       --
 *                   p2                     p2
 *
 */
__ARMAS_INLINE
void __pivot_repart_2x1to3x1(armas_pivot_t *pT, armas_pivot_t *p0,
			     armas_pivot_t *p1, armas_pivot_t *p2,
			     armas_pivot_t *P, int nb, int direction)
{
  int nT = armas_pivot_size(pT);
  switch (direction) {
  case ARMAS_PBOTTOM:
    if (nT + nb > armas_pivot_size(P)) {
      nb = armas_pivot_size(P) - nT;
    }
    __subpivot(p0, P, 0,  nT);
    __subpivot(p1, P, nT, nb);
    __subpivot(p2, P, nT+nb, armas_pivot_size(P)-nT-nb);
    break;
  case ARMAS_PTOP:
    if (nb > nT) {
      nb = nT;
    }
    __subpivot(p0, P, 0, nT-nb);
    __subpivot(p1, P, nT-nb, nb);
    __subpivot(p2, P, nT, armas_pivot_size(P)-nT);
    break;
  }
}


/*
 * Continue with 2 by 1 block from 3 by 1 block.
 * 
 *           pT      p0            pT       p0
 * pBOTTOM: --  <--  p1   ; pTOP:   -- <--  --
 *           pB      --            pB       p1
 *                   p2                     p2
 *
 */
__ARMAS_INLINE
void __pivot_cont_3x1to2x1(armas_pivot_t *pT, armas_pivot_t *pB,
			   armas_pivot_t *p0, armas_pivot_t *p1,
			   armas_pivot_t *P, int direction)
{
  int n0 = armas_pivot_size(p0);
  int n1 = armas_pivot_size(p1);
  switch (direction) {
  case ARMAS_PBOTTOM:
    __subpivot(pT, P, 0, n0+n1);
    __subpivot(pB, P, n0+n1, armas_pivot_size(P)-n0-n1);
    break;
  case ARMAS_PTOP:
    __subpivot(pT, P, 0, n0);
    __subpivot(pB, P, n0, armas_pivot_size(P)-n0);
    break;
  }
}

#endif /* __PIVOT_H */
