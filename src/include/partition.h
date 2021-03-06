
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef ARMAS_PARTITION_H
#define ARMAS_PARTITION_H 1


#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE extern inline
#endif

#include "matrix.h"

//! @addtogroup internal
//! @{

/**
 * @brief Partition A to 2 by 1 blocks.
 *
 * Partition
 *  \f$
 *      A
 *         \overset{nb}\rightarrow
 *      \begin{pmatrix} A_T\\ \hline A_B \end{pmatrix}
 *   \f$
 *
 * @param[out] AT, AB Result blocks
 * @param[in]  A      Source matrix
 * @param[in]  nb     Initial block size for AT or AB
 * @param[in]  side   Starting side, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void mat_partition_2x1(armas_dense_t *AT, armas_dense_t *AB,
                       const armas_dense_t *A, int nb, int side)
{
    if (nb > A->rows)
        nb = A->rows;

    switch (side) {
    case ARMAS_PTOP:
        armas_submatrix_unsafe(AT, A, 0,  0, nb, A->cols);
        armas_submatrix_unsafe(AB, A, nb, 0, A->rows-nb, A->cols);
        break;
    case ARMAS_PBOTTOM:
    default:
        armas_submatrix_unsafe(AT, A, 0, 0, A->rows-nb,  A->cols);
        armas_submatrix_unsafe(AB, A, A->rows-nb, 0, nb, A->cols);
        break;
    }
}

/**
 * @brief Repartition 2 by 1 block to 3 by 1 block.
 *
 * Repartition
 * *ARMAS_PTOP*:
 *  \f$
 *     \begin{pmatrix} A_T \\ \hline A_B \end{pmatrix}
 *        \overset{nb}\rightarrow
 *     \begin{pmatrix} A_0 \\ A_1 \\ \hline A_2 \end{pmatrix}
 *  , \quad \f$
 * *ARMAS_PBOTTOM*:
 *  \f$
 *     \begin{pmatrix} A_T \\ \hline A_B \end{pmatrix}
 *        \overset{nb}\rightarrow
 *     \begin{pmatrix} A_0 \\ \hline A_1 \\ A_2 \end{pmatrix}
 *  \f$
 *
 * @param[in]  AT          Top bloack
 * @param[out] A0, A1, A2  Result blocks
 * @param[in]  A           Source matrix
 * @param[in]  nb          Block size for A1
 * @param[in]  direction   Blocking direction, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void mat_repartition_2x1to3x1(armas_dense_t *AT, armas_dense_t *A0,
                              armas_dense_t *A1, armas_dense_t *A2,
                              const armas_dense_t *A, int nb, int direction)
{
    switch (direction) {
    case ARMAS_PBOTTOM:
        if (nb + AT->rows > A->rows) {
            nb = A->rows - AT->rows;
        }
        armas_submatrix_unsafe(A0, A, 0, 0, AT->rows, A->cols);
        armas_submatrix_unsafe(A1, A, AT->rows, 0, nb, A->cols);
        armas_submatrix_unsafe(A2, A, AT->rows+nb, 0, A->rows-AT->rows-nb, A->cols);
        break;
    case ARMAS_PTOP:
    default:
        if (AT->rows < nb) {
            nb = AT->rows;
        }
        armas_submatrix_unsafe(A0, A, 0, 0, AT->rows-nb,  A->cols);
        armas_submatrix_unsafe(A1, A, AT->rows-nb, 0, nb, A->cols);
        armas_submatrix_unsafe(A2, A, AT->rows, 0, A->rows-AT->rows, A->cols);
        break;
    }
}

/**
 * @brief Continue with 2 by 1 block from 3 by 1 block.
 *
 * *ARMAS_PTOP*:
 *  \f$
 *     \begin{pmatrix} A_0 \\ \hline A_1 \\ A_2 \end{pmatrix}
 *        \rightarrow
 *     \begin{pmatrix} A_T \\ \hline A_B \end{pmatrix}
 *  , \quad \f$
 * *ARMAS_PBOTTOM*:
 *  \f$
 *     \begin{pmatrix} A_0 \\ A_1 \\ \hline A_2 \end{pmatrix}
 *        \rightarrow
 *     \begin{pmatrix} A_T \\ \hline A_B \end{pmatrix}
 *  \f$
 *
 * @param[out] AT, AB      Result blocks
 * @param[in]  A0, A1      Source blocks
 * @param[in]  A           Source matrix
 * @param[in]  direction   Blocking direction, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void mat_continue_3x1to2x1(armas_dense_t *AT, armas_dense_t *AB,
                           armas_dense_t *A0, armas_dense_t *A1,
                           const armas_dense_t *A, int direction)
{
    int nr0 = A0->rows;
    int nr1 = A1->rows;
    switch (direction) {
    case ARMAS_PBOTTOM:
        armas_submatrix_unsafe(AT, A, 0,       0, nr0+nr1, A->cols);
        armas_submatrix_unsafe(AB, A, nr0+nr1, 0, A->rows-nr0-nr1, A->cols);
        break;
    case ARMAS_PTOP:
    default:
        armas_submatrix_unsafe(AT, A, 0, 0,   nr0,  A->cols);
        armas_submatrix_unsafe(AB, A, nr0, 0, A->rows-nr0, A->cols);
        break;
    }
}

/**
 * @brief Partition A to 1 by 2 blocks.
 * \f$
 *   A
 *     \overset{nb}\rightarrow
 *   \begin{pmatrix}\begin{array}{c|c} A_L & A_R \end{array}\end{pmatrix}
 * \f$
 *
 * @cond
 *  A -->  AL | AR
 * @endcond
 * @param[out] AL, AR Result blocks
 * @param[in]  A      Source matrix
 * @param[in]  nb     Initial block size for AL or AR
 * @param[in]  side   Starting side, ARMAS_PLEFT or ARMAS_PRIGHT
 */
__ARMAS_INLINE
void mat_partition_1x2(armas_dense_t *AL, armas_dense_t *AR,
                       const armas_dense_t *A, int nb, int side)
{
    if (nb > A->cols) {
        nb = A->cols;
    }
    switch (side) {
    case ARMAS_PLEFT:
        armas_submatrix_unsafe(AL, A, 0, 0,  A->rows, nb);
        armas_submatrix_unsafe(AR, A, 0, nb, A->rows, A->cols-nb);
        break;
    case ARMAS_PRIGHT:
    default:
        armas_submatrix_unsafe(AL, A, 0, 0,  A->rows, A->cols-nb);
        armas_submatrix_unsafe(AR, A, 0, A->cols-nb,  A->rows, nb);
        break;
    }
}

/**
 * @brief Repartition 1 by 2 blocks to 1 by 3 blocks.
 *
 * *ARMAS_PLEFT* \f$
 *   \begin{pmatrix}\begin{array}{c|c}
 *     A_L & A_R
 *   \end{array}\end{pmatrix}
 *      \overset{nb}\rightarrow
 *   \begin{pmatrix}\begin{array}{cc|c}
 *     A_0 & A_1 & A_2
 *   \end{array}\end{pmatrix} , \quad
 * \f$
 * *ARMAS_PRIGHT* \f$
 *   \begin{pmatrix}\begin{array}{c|c}
 *     A_L & A_R
 *   \end{array}\end{pmatrix}
 *      \overset{nb}\rightarrow
 *   \begin{pmatrix}\begin{array}{c|cc}
 *     A_0 & A_1 & A_2
 *   \end{array}\end{pmatrix}
 * \f$
 * @cond
 *```txt
 * pRIGHT: AL | AR  -->  A0 | A1 A2
 * pLEFT:  AL | AR  -->  A0 A1 | A2
 *```
 * @endcond
 * @param[in]  AL          Left block
 * @param[out] A0, A1, A2  Result blocks
 * @param[in]  A           Source matrix
 * @param[in]  nb          Block size for A1
 * @param[in]  direction   Blocking direction, ARMAS_PLEFT or ARMAS_PRIGHT.
 */
__ARMAS_INLINE
void mat_repartition_1x2to1x3(armas_dense_t *AL, armas_dense_t *A0,
                              armas_dense_t *A1, armas_dense_t *A2,
                              const armas_dense_t *A, int nb, int direction)
{
    switch (direction) {
    case ARMAS_PLEFT:
        if (nb > AL->cols) {
            nb = AL->cols;
        }
        armas_submatrix_unsafe(A0, A, 0, 0, A->rows,  AL->cols-nb);
        armas_submatrix_unsafe(A1, A, 0, AL->cols-nb, A->rows, nb);
        armas_submatrix_unsafe(A2, A, 0, AL->cols,    A->rows, A->cols-AL->cols);
        break;
    case ARMAS_PRIGHT:
    default:
        if (AL->cols + nb > A->cols) {
            nb = A->cols - AL->cols;
        }
        armas_submatrix_unsafe(A0, A, 0, 0, A->rows, AL->cols);
        armas_submatrix_unsafe(A1, A, 0, AL->cols,    A->rows, nb);
        armas_submatrix_unsafe(A2, A, 0, AL->cols+nb, A->rows, A->cols-AL->cols-nb);
        break;
    }
}

/**
 * @brief Redefine 1 by 2 blocks from 1 by 3 blocks.
 *
 * *ARMAS_PLEFT* \f$
 *   \begin{pmatrix}\begin{array}{c|cc}
 *     A_0 & A_1 & A_2
 *   \end{array}\end{pmatrix}
 *      \rightarrow
 *   \begin{pmatrix}\begin{array}{c|c}
 *     A_L & A_R
 *   \end{array}\end{pmatrix}
 *    , \quad
 * \f$
 * *ARMAS_PRIGHT* \f$
 *   \begin{pmatrix}\begin{array}{cc|c}
 *     A_0 & A_1 & A_2
 *   \end{array}\end{pmatrix}
 *      \rightarrow
 *   \begin{pmatrix}\begin{array}{c|c}
 *     A_L & A_R
 *   \end{array}\end{pmatrix}
 * \f$
 * @cond
 * pRIGHT: AL | AR  --  A0 A1 | A2
 * pLEFT:  AL | AR  <--  A0 | A1 A2
 * @endcond
 *
 * @param[out] AL, AR      Result blocks
 * @param[in]  A0, A1      Source blocks
 * @param[in]  A           Source matrix
 * @param[in]  direction   Blocking direction, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void mat_continue_1x3to1x2(armas_dense_t *AL, armas_dense_t *AR,
                           armas_dense_t *A0, armas_dense_t *A1,
                           const armas_dense_t *A, int direction)
{
    int nl;
    switch (direction) {
    case ARMAS_PLEFT:
        armas_submatrix_unsafe(AL, A, 0, 0, A->rows, A0->cols);
        armas_submatrix_unsafe(AR, A, 0, A0->cols, A->rows, A->cols-A0->cols);
        break;
    case ARMAS_PRIGHT:
        nl = A0->cols + A1->cols;
        armas_submatrix_unsafe(AL, A, 0, 0, A->rows, nl);
        armas_submatrix_unsafe(AR, A, 0, nl, A->rows, A->cols-nl);
        break;
    }
}

/**
 * @brief Partition A to 2 by 2 blocks.
 *
 * \f$
 *     A
 *       \overset{mb,nb}\rightarrow
 *    \begin{pmatrix}\begin{array}{c|c}
 *       A_{TL} & A_{TR} \\
 *       \hline
 *       A_{BL} & A_{BR}
 *    \end{array}\end{pmatrix}
 * \f$
 * @param[out] ATL, ATR, ABL, ABR Result blocks
 * @param[in]  A      Source matrix
 * @param[in]  mb     Rows in initial block ATL or ABR
 * @param[in]  nb     Columns in initial block ATL or ABR
 * @param[in]  side   Starting side, ARMAS_PTOPLEFT or ARMAS_PBOTTOMRIGHT.
 *
 * Note: ATR and ABL matrices may be null pointers.
 */
__ARMAS_INLINE
void mat_partition_2x2(armas_dense_t *ATL, armas_dense_t *ATR,
                       armas_dense_t *ABL, armas_dense_t *ABR,
                       const armas_dense_t *A, int mb, int nb, int side)
{
    switch (side) {
    case ARMAS_PTOPLEFT:
        armas_submatrix_unsafe(ATL, A, 0, 0, mb, nb);
        if (ATR)
            armas_submatrix_unsafe(ATR, A, 0, nb, mb, A->cols-nb);
        if (ABL)
            armas_submatrix_unsafe(ABL, A, mb, 0, A->rows-mb, nb);
        armas_submatrix_unsafe(ABR, A, mb, nb, A->rows-mb, A->cols-nb);
        break;
    case ARMAS_PBOTTOMRIGHT:
    default:
        armas_submatrix_unsafe(ATL, A, 0, 0, A->rows-mb, A->cols-nb);
        if (ATR)
            armas_submatrix_unsafe(ATR, A, 0, A->cols-nb, A->rows-mb, nb);
        if (ABL)
            armas_submatrix_unsafe(ABL, A, A->rows-mb, 0, mb, A->cols-nb);
        armas_submatrix_unsafe(ABR, A, A->rows-mb, A->cols-nb, mb, nb);
        break;
    }
}

/**
 * @brief Repartition 2 by 2 blocks to 3 by 3 blocks.
 *
 * \f$
 *      \begin{pmatrix}\begin{array}{c|c}
 *      A_{TL} & A_{TR} \\
 *      \hline
 *      A_{BL} & A_{BR}
 *      \end{array}\end{pmatrix}
 *         \overset{nb}\rightarrow
 *      \begin{pmatrix}\begin{array}{c|cc}
 *      A_{00} & A_{01} & A_{02}\\
 *      \hline
 *      A_{10} & A_{11} & A_{12}\\
 *      A_{20} & A_{21} & A_{22}\\
 *      \end{array}\end{pmatrix}
 * \f$
 * @cond
 *                      A00 | A01 : A02
 *   ATL | ATR   nb     ===============
 *   =========   -->    A10 | A11 : A12
 *   ABL | ABR          ---------------
 *                      A20 | A21 : A22
 * @endcond
 *
 * @param[in]  ATL           Top block
 * @param[out] A00, A01, A02 Result block (may be null pointer except A00)
 * @param[out] A10, A11, A12 Result block (may be null pointer except A11)
 * @param[out] A20, A21, A22 Result block (may be null pointer except A22)
 * @param[in]  A             Source matrix
 * @param[in]  nb            Size of A11
 * @param[in]  direction     Blocking direction ARMAS_PTOPLEFT or ARMAS_PBOTTOMRIGHT.
 *
 * Blocks ATR, ABL, ABR implicitely defined by ATL and A.
 * All but diagonal matrices may be null pointers.
 */
__ARMAS_INLINE
void mat_repartition_2x2to3x3(armas_dense_t *ATL,
                              armas_dense_t *A00, armas_dense_t *A01, armas_dense_t *A02,
                              armas_dense_t *A10, armas_dense_t *A11, armas_dense_t *A12,
                              armas_dense_t *A20, armas_dense_t *A21, armas_dense_t *A22,
                              const armas_dense_t *A, int nb, int direction)
{
    int kr = ATL->rows;
    int kc = ATL->cols;
    switch (direction) {
    case ARMAS_PBOTTOMRIGHT:
        if (kc + nb > A->cols) {
            nb = A->cols - kc;
        }
        if (kr + nb > A->rows) {
            nb = A->rows - kr;
        }
        armas_submatrix_unsafe(A00, A, 0, 0, kr, kc);
        if (A01)
            armas_submatrix_unsafe(A01, A, 0, kc, kr, nb);
        if (A02)
            armas_submatrix_unsafe(A02, A, 0, kc+nb, kr, A->cols-kc-nb);
        if (A10)
            armas_submatrix_unsafe(A10, A, kr, 0, nb, kc);
        armas_submatrix_unsafe(A11, A, kr, kc, nb, nb);
        if (A12)
            armas_submatrix_unsafe(A12, A, kr, kc+nb, nb, A->cols-kc-nb);
        if (A20)
            armas_submatrix_unsafe(A20, A, kr+nb, 0, A->rows-kr-nb, kc);
        if (A21)
            armas_submatrix_unsafe(A21, A, kr+nb, kc, A->rows-kr-nb, nb);
        armas_submatrix_unsafe(A22, A, kr+nb, kc+nb, A->rows-kr-nb, A->cols-kc-nb);
        break;
    case ARMAS_PTOPLEFT:
    default:
        // move towards top left corner
        if (nb > kc)
            nb = kc;
        if (nb > kr)
            nb = kr;
        armas_submatrix_unsafe(A00, A, 0, 0, kr-nb, kc-nb);
        if (A01)
            armas_submatrix_unsafe(A01, A, 0, kc-nb, kr-nb, nb);
        if (A02)
            armas_submatrix_unsafe(A02, A, 0, kc, kr-nb, A->cols-kc);
        if (A10)
            armas_submatrix_unsafe(A10, A, kr-nb, 0, nb, kc-nb);
        armas_submatrix_unsafe(A11, A, kr-nb, kc-nb, nb, nb);
        if (A12)
            armas_submatrix_unsafe(A12, A, kr-nb, kc, nb, A->cols-kc);
        if (A20)
            armas_submatrix_unsafe(A20, A, kr, 0, A->rows-kr, kc-nb);
        if (A21)
            armas_submatrix_unsafe(A21, A, kr, kc-nb, A->rows-kr, nb);
        armas_submatrix_unsafe(A22, A, kr, kc, A->rows-kr, A->cols-kc);
        break;
    }
}

/**
 * @brief Redefine 2 by 2 blocks from 3 by 3 partition.
 *
 * \f$
 *      \begin{pmatrix}\begin{array}{cc|c}
 *      A_{00} & A_{01} & A_{02}\\
 *      A_{10} & A_{11} & A_{12}\\
 *      \hline
 *      A_{20} & A_{21} & A_{22}\\
 *      \end{array}\end{pmatrix}
 *          \rightarrow
 *      \begin{pmatrix}\begin{array}{c|c}
 *      A_{TL} & A_{TR} \\
 *      \hline
 *      A_{BL} & A_{BR}
 *      \end{array}\end{pmatrix}
 * \f$
 * @cond
 *                      A00 : A01 | A02
 *   ATL | ATR   nb     ---------------
 *   =========   <--    A10 : A11 | A12
 *   ABL | ABR          ===============
 *                      A20 : A21 | A22
 * @endcond
 *
 * @param[out] ATL, ATR, ABL, ABR Result blocks
 * @param[in]  A00, A11, A22      Rows in initial block ATL or ABR
 * @param[in]  A                  Source matrix
 * @param[in]  direction          ARMAS_PTOPLEFT or ARMAS_PBOTTOMRIGHT.
 *
 * Note: ATR and ABL matrices may be null pointers.
 * New division of ATL, ATR, ABL, ABR defined by diagonal entries A00, A11, A22
 */
__ARMAS_INLINE
void mat_continue_3x3to2x2(armas_dense_t *ATL, armas_dense_t *ATR,
                           armas_dense_t *ABL, armas_dense_t *ABR,
                           armas_dense_t *A00, armas_dense_t *A11, armas_dense_t *A22,
                           const armas_dense_t *A, int direction)
{
    //int nk = A00->rows;
    int mb = A11->cols;
    int kr = A00->rows;
    int kc = A00->cols;
    switch (direction) {
    case ARMAS_PBOTTOMRIGHT:
        armas_submatrix_unsafe(ATL, A, 0, 0,     kr+mb, kc+mb);
        if (ATR)
            armas_submatrix_unsafe(ATR, A, 0, kc+mb, kr+mb, A->cols-kc-mb);
        if (ABL)
            armas_submatrix_unsafe(ABL, A, kr+mb, 0, A->rows-kr-mb, kc+mb);
        armas_submatrix_unsafe(ABR, A, kr+mb, kc+mb, A->rows-kr-mb, A->cols-kc-mb);
        break;
    case ARMAS_PTOPLEFT:
    default:
        armas_submatrix_unsafe(ATL, A, 0, 0,  kr, kc);
        if (ATR)
            armas_submatrix_unsafe(ATR, A, 0, kc, kr, A->cols-kc);
        if (ABL)
            armas_submatrix_unsafe(ABL, A, kr, 0, A->rows-kr, A->cols-kc);
        armas_submatrix_unsafe(ABR, A, kr, kc, A->rows-kr, A->cols-kc);
        break;
    }
}

/**
 * @brief Merge 1 by 1 block from 2 by 1 block.
 *
 * Merge block from two adjacent blocks.
 *  \f$
 *    \begin{pmatrix} A_T\\ \hline A_B \end{pmatrix} \rightarrow A_{blk}
 *  \f$
 *
 * @param[out] ABLK   Result block
 * @param[in]  AT, AB Source blocks
 */
__ARMAS_INLINE
void mat_merge2x1(armas_dense_t *ABLK, armas_dense_t *AT, armas_dense_t *AB)
{
    require(AT->step == AB->step);
    if (armas_size(AT) == 0 && armas_size(AB) == 0) {
        ABLK->rows = 0; ABLK->cols = 0; ABLK->step = AB->step;
        return;
    }
    if (armas_size(AT) == 0) {
        armas_submatrix_unsafe(ABLK, AB, 0, 0, AB->rows, AB->cols);
    } else if (armas_size(AB) == 0) {
        armas_submatrix_unsafe(ABLK, AT, 0, 0, AT->rows, AT->cols);
    } else {
        armas_make(ABLK, AT->rows+AB->rows, AT->cols, AT->step, AT->elems);
    }
}

/**
 * @brief Merge 1 by 1 block from 1 by 2 block.
 *
 * Merge block from two adjacent blocks.
 *   \f$
 *       \begin{pmatrix}\begin{array}{c|c} A_L & A_R \end{array}\end{pmatrix} \rightarrow  A_{blk}
 *   \f$
 * @cond
 * ABLK <--  AL | AR
 * @endcond
 * @param[out] ABLK   Result block
 * @param[in]  AL, AR Source blocks
 */
__ARMAS_INLINE
void mat_merge1x2(armas_dense_t *ABLK, armas_dense_t *AL, armas_dense_t *AR)
{
    require(AL->step == AR->step);
    if (armas_size(AL) == 0 && armas_size(AR) == 0) {
        ABLK->rows = 0; ABLK->cols = 0; ABLK->step = AL->step;
        return;
    }
    if (armas_size(AL) == 0) {
        armas_submatrix_unsafe(ABLK, AR, 0, 0, AR->rows, AR->cols);
    } else if (armas_size(AR) == 0) {
        armas_submatrix_unsafe(ABLK, AL, 0, 0, AL->rows, AL->cols);
    } else {
        armas_make(ABLK, AL->rows, AL->cols+AR->cols, AL->step, AL->elems);
    }
}

/**
 * @brief Partition x to 2 by 1 blocks.
 *
 * Partition
 *  \f$
 *    x
 *      \overset{nb}\rightarrow
 *    \begin{pmatrix} x_T\\ \hline x_B \end{pmatrix}
 *  \f$
 *
 * @param[out] xT, xB Result subvector
 * @param[in]  x      Source vector
 * @param[in]  nb     Initial length of xT or xB
 * @param[in]  side   Starting side, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void vec_partition_2x1(armas_dense_t *xT, armas_dense_t *xB,
                       const armas_dense_t *x, int nb, int side)
{
    int nr = armas_size(x);
    if (nb > nr)
        nb = nr;

    switch (side) {
    case ARMAS_PTOP:
        armas_subvector_unsafe(xT, x, 0, nb);
        armas_subvector_unsafe(xB, x, nb, nr-nb);
        break;
    case ARMAS_PBOTTOM:
    default:
        armas_subvector_unsafe(xT, x, 0, nr-nb);
        armas_subvector_unsafe(xB, x, nr-nb, nb);
        break;
    }
}

/**
 * @brief Repartition 2 by 1 block to 3 by 1 block.
 *
 * *ARMAS_PTOP*:
 *  \f$
 *     \begin{pmatrix} x_T \\ \hline x_B \end{pmatrix}
 *       \overset{nb}\rightarrow
 *     \begin{pmatrix} x_0 \\ x_1 \\ \hline x_2 \end{pmatrix}
 *  , \quad
 *  \f$
 * *ARMAS_PBOTTOM*:
 *  \f$
 *     \begin{pmatrix} x_T \\ \hline x_B \end{pmatrix}
 *        \overset{nb}\rightarrow
 *     \begin{pmatrix} x_0 \\ \hline x_1 \\ x_2 \end{pmatrix}
 *  \f$
 * @cond
 *           xT      x0            xT       x0
 * pBOTTOM: --  --> --   ; pTOP:   --  -->  x1
 *           xB      x1            xB       --
 *                   x2                     x2
 * @endcond
 *
 * @param[in]  xT          Top subvector
 * @param[out] x0, x1, x2  Result subvectors
 * @param[in]  x           Source vector
 * @param[in]  nb          Length of x1
 * @param[in]  direction   Blocking direction, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void vec_repartition_2x1to3x1(armas_dense_t *xT, armas_dense_t *x0,
                              armas_dense_t *x1, armas_dense_t *x2,
                              const armas_dense_t *x, int nb, int direction)
{
    int nr = armas_size(x);
    int nrt = armas_size(xT);
    switch (direction) {
    case ARMAS_PBOTTOM:
        if (nb + nrt > nr) {
            nb = nr - nrt;
        }
        armas_subvector_unsafe(x0, x, 0, nrt);
        armas_subvector_unsafe(x1, x, nrt, nb);
        armas_subvector_unsafe(x2, x, nrt+nb, nr-nrt-nb);
        break;
    case ARMAS_PTOP:
    default:
        if (nrt < nb) {
            nb = nrt;
        }
        armas_subvector_unsafe(x0, x, 0, nrt-nb);
        armas_subvector_unsafe(x1, x, nrt-nb, nb);
        armas_subvector_unsafe(x2, x, nrt, nr-nrt);
        break;
    }
}

/**
 * @brief Continue with 2 by 1 block from 3 by 1 block.
 *
 * *ARMAS_PTOP*:
 *  \f$
 *     \begin{pmatrix} x_0 \\ \hline x_1 \\ x_2 \end{pmatrix}
 *       \rightarrow
 *     \begin{pmatrix} x_T \\ \hline x_B \end{pmatrix}
 *  , \quad
 *  \f$
 * *ARMAS_PBOTTOM*:
 *  \f$
 *     \begin{pmatrix} x_0 \\ x_1 \\ \hline x_2 \end{pmatrix}
 *        \rightarrow
 *     \begin{pmatrix} x_T \\ \hline x_B \end{pmatrix}
 *  \f$
 * @cond
 *           xT      x0            xT       x0
 * pBOTTOM: --  <--  x1   ; pTOP:   -- <--  --
 *           xB      --            xB       x1
 *                   x2                     x2
 * @endcond
 *
 * @param[out] xT, xB     Result subvectors
 * @param[in]  x0, x1     Source subvectors
 * @param[in]  x          Source vector
 * @param[in]  direction  Blocking direction, ARMAS_PTOP or ARMAS_PBOTTOM.
 */
__ARMAS_INLINE
void vec_continue_3x1to2x1(armas_dense_t *xT, armas_dense_t *xB,
                           armas_dense_t *x0, armas_dense_t *x1,
                           const armas_dense_t *x, int direction)
{
    int nr = armas_size(x);
    int nr0 = armas_size(x0);
    int nr1 = armas_size(x1);
    switch (direction) {
    case ARMAS_PBOTTOM:
        armas_subvector_unsafe(xT, x, 0, nr0+nr1);
        armas_subvector_unsafe(xB, x, nr0+nr1, nr-nr0-nr1);
        break;
    case ARMAS_PTOP:
    default:
        armas_subvector_unsafe(xT, x, 0, nr0);
        armas_subvector_unsafe(xB, x, nr0, nr-nr0);
        break;
    }
}
//! @}
#endif
