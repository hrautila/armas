

#ifndef __PIVOT_H
#define __PIVOT_H 1

#ifndef __INLINE
#define __INLINE extern inline
#endif

static inline
void __subpivot(armas_pivots_t *pA, armas_pivots_t *pB, int K, int N)
{
  pA->npivots = N;
  pA->indexes = &pB->indexes[K];
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
__INLINE
void __armas_pivot_2x1(armas_pivots_t *pT, armas_pivots_t *pB,
                       armas_pivots_t *P, int nb, int direction)
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
__INLINE
void __armas_pivot_repart_2x1to3x1(armas_pivots_t *pT, armas_pivots_t *p0,
                                   armas_pivots_t *p1, armas_pivots_t *p2,
                                   armas_pivots_t *P, int nb, int direction)
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
__INLINE
void __armas_pivot_cont_3x1to2x1(armas_pivots_t *pT, armas_pivots_t *pB,
                                 armas_pivots_t *p0, armas_pivots_t *p1,
                                 armas_pivots_t *P, int direction)
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
