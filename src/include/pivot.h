
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.



#ifndef ARMAS_PIVOT_H
#define ARMAS_PIVOT_H 1

#include "armas.h"

#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE extern inline
#endif

//! @addtogroup internal
//! @{

static inline
void subpivot(armas_pivot_t *pA, armas_pivot_t *pB, int K, int N)
{
    pA->npivots = N;
    pA->indexes = &pB->indexes[K];
    pA->owner = 0;
}

/**
 * @brief Partition p to 2 by 1 blocks.
 *
 * Partition
 *  \f$
 *     P
 *      \overset{nb}\rightarrow
 *     \begin{pmatrix} p_T\\ \hline p_B \end{pmatrix}
 *  \f$
 *
 * @param[out] pT, pB
 *   Result index vectors
 * @param[in]  P
 *   Pivot index vector
 * @param[in]  nb
 *   Initial vector length
 * @param[in]  side
 *   Start of partitioning ARMAS_PTOP or ARMAS_PBOTTOM
 */
static inline
void pivot_2x1(armas_pivot_t *pT, armas_pivot_t *pB, armas_pivot_t *P, int nb, int side)
{
    switch (side) {
    case ARMAS_PTOP:
        if (nb > P->npivots)
            nb = P->npivots;
        subpivot(pT, P, 0, nb);
        subpivot(pB, P, nb, P->npivots-nb);
        break;
    case ARMAS_PBOTTOM:
        subpivot(pT, P, 0,  P->npivots-nb);
        subpivot(pB, P, P->npivots-nb, nb);
        break;
    }
}


/**
 * @brief Repartition 2 by 1 block to 3 by 1 block.
 *
 * *ARMAS_PTOP*:
 *  \f$
 *     \begin{pmatrix} p_T \\ \hline p_B \end{pmatrix}
 *       \overset{nb}\rightarrow
 *     \begin{pmatrix} p_0 \\ p_1 \\ \hline p_2 \end{pmatrix}
 *  , \quad
 *  \f$
 * *ARMAS_PBOTTOM*:
 *  \f$
 *     \begin{pmatrix} p_T \\ \hline p_B \end{pmatrix}
 *       \overset{nb}\rightarrow
 *     \begin{pmatrix} p_0 \\ \hline p_1 \\ p_2 \end{pmatrix}
 *  \f$
 *
 * @param[in]  pT
 *  Topmost pivot vector
 * @param[out] p0, p1, p2
 *   Result pivot subvectors
 * @param[in]  P
 *   Pivot index vector
 * @param[in]  nb
 *   Length of p1 subvector
 * @param[in]  direction
 *   Partitioning direction ARMAS_PTOP or ARMAS_PBOTTOM
 */
static inline
void pivot_repart_2x1to3x1(
    armas_pivot_t *pT, armas_pivot_t *p0, armas_pivot_t *p1, armas_pivot_t *p2, armas_pivot_t *P, int nb, int direction)
{
    int nT = armas_pivot_size(pT);
    switch (direction) {
    case ARMAS_PBOTTOM:
        if (nT + nb > armas_pivot_size(P)) {
            nb = armas_pivot_size(P) - nT;
        }
        subpivot(p0, P, 0,  nT);
        subpivot(p1, P, nT, nb);
        subpivot(p2, P, nT+nb, armas_pivot_size(P)-nT-nb);
        break;
    case ARMAS_PTOP:
        if (nb > nT) {
            nb = nT;
        }
        subpivot(p0, P, 0, nT-nb);
        subpivot(p1, P, nT-nb, nb);
        subpivot(p2, P, nT, armas_pivot_size(P)-nT);
        break;
    }
}


/**
 * @brief Continue with 2 by 1 block from 3 by 1 block.
 *
 * *ARMAS_PTOP*:
 *  \f$
 *     \begin{pmatrix} p_0 \\ \hline p_1 \\ p_2 \end{pmatrix}
 *       \rightarrow
 *     \begin{pmatrix} p_T \\ \hline p_B \end{pmatrix}
 *  , \quad \f$
 * *ARMAS_PBOTTOM*:
 *  \f$
 *     \begin{pmatrix} p_0 \\ p_1 \\ \hline p_2 \end{pmatrix}
 *       \rightarrow
 *     \begin{pmatrix} p_T \\ \hline p_B \end{pmatrix}
 *  \f$
 *
 * @param[out] pT, pB
 *   Result pivot vectors
 * @param[in]  p0, p1
 *   Pivot subvectors
 * @param[in]  P
 *   Pivot index vector
 * @param[in]  direction
 *   Partitioning direcion, ARMAS_PTOP or ARMAS_PBOTTOM
 */
static inline
void pivot_cont_3x1to2x1(
    armas_pivot_t *pT, armas_pivot_t *pB, armas_pivot_t *p0, armas_pivot_t *p1, armas_pivot_t *P, int direction)
{
    int n0 = armas_pivot_size(p0);
    int n1 = armas_pivot_size(p1);
    switch (direction) {
    case ARMAS_PBOTTOM:
        subpivot(pT, P, 0, n0+n1);
        subpivot(pB, P, n0+n1, armas_pivot_size(P)-n0-n1);
        break;
    case ARMAS_PTOP:
        subpivot(pT, P, 0, n0);
        subpivot(pB, P, n0, armas_pivot_size(P)-n0);
        break;
    }
}
//! @}
#endif /* PIVOT_H */
