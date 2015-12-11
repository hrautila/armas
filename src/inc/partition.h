
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#ifndef __ARMAS_PARTITION_H
#define __ARMAS_PARTITION_H 1


#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE extern inline
#endif


/*
 * Partition p to 2 by 1 blocks.
 *
 *        AT
 *  A --> --
 *        AB
 *
 * Parameter nb is initial block size for AT (PTOP) or AB (PBOTTOM).  
 */
__ARMAS_INLINE
void __partition_2x1(__armas_dense_t *AT, __armas_dense_t *AB,
                     const __armas_dense_t *A, int nb, int side)
{
  if (nb > A->rows)
    nb = A->rows;

  switch (side) {
  case ARMAS_PTOP:
    __armas_submatrix_unsafe(AT, A, 0,  0, nb, A->cols);
    __armas_submatrix_unsafe(AB, A, nb, 0, A->rows-nb, A->cols);
    break;
  case ARMAS_PBOTTOM:
  default:
    __armas_submatrix_unsafe(AT, A, 0, 0, A->rows-nb,  A->cols);
    __armas_submatrix_unsafe(AB, A, A->rows-nb, 0, nb, A->cols);
    break;
  }
}
			
/*
 * Repartition 2 by 1 block to 3 by 1 block.
 * 
 *           AT      A0            AT       A0
 * pBOTTOM: --  --> --   ; pTOP:   --  -->  A1
 *           AB      A1            AB       --
 *                   A2                     A2
 *
 */
__ARMAS_INLINE
void __repartition_2x1to3x1(__armas_dense_t *AT, __armas_dense_t *A0,
                            __armas_dense_t *A1, __armas_dense_t *A2,
                            const __armas_dense_t *A, int nb, int direction)
{
  switch (direction) {
  case ARMAS_PBOTTOM:
    if (nb + AT->rows > A->rows) {
      nb = A->rows - AT->rows;
    }
    __armas_submatrix_unsafe(A0, A, 0, 0, AT->rows, A->cols);
    __armas_submatrix_unsafe(A1, A, AT->rows, 0, nb, A->cols);
    __armas_submatrix_unsafe(A2, A, AT->rows+nb, 0, A->rows-AT->rows-nb, A->cols);
    break;
  case ARMAS_PTOP:
  default:
    if (AT->rows < nb) {
      nb = AT->rows;
    }
    __armas_submatrix_unsafe(A0, A, 0, 0, AT->rows-nb,  A->cols);
    __armas_submatrix_unsafe(A1, A, AT->rows-nb, 0, nb, A->cols);
    __armas_submatrix_unsafe(A2, A, AT->rows, 0, A->rows-AT->rows, A->cols);
    break;
  }
}

/*
 * Continue with 2 by 1 block from 3 by 1 block.
 * 
 *           AT      A0            AT       A0
 * pBOTTOM: --  <--  A1   ; pTOP:   -- <--  --
 *           AB      --            AB       A1
 *                   A2                     A2
 */
__ARMAS_INLINE
void __continue_3x1to2x1(__armas_dense_t *AT, __armas_dense_t *AB,
                         __armas_dense_t *A0, __armas_dense_t *A1,
                         const __armas_dense_t *A, int direction)
{
  int nr0 = A0->rows;
  int nr1 = A1->rows;
  switch (direction) {
  case ARMAS_PBOTTOM:
    __armas_submatrix(AT, A, 0,       0, nr0+nr1, A->cols);
    __armas_submatrix(AB, A, nr0+nr1, 0, A->rows-nr0-nr1, A->cols);
    break;
  case ARMAS_PTOP:
  default:
    __armas_submatrix(AT, A, 0, 0,   nr0,  A->cols);
    __armas_submatrix(AB, A, nr0, 0, A->rows-nr0, A->cols);
    break;
  }
}

/*
 * Partition A to 1 by 2 blocks.
 *
 *  A -->  AL | AR
 *
 * Parameter nb is initial block size for AL (PLEFT) or AR (PRIGHT).  
 */
__ARMAS_INLINE
void __partition_1x2(__armas_dense_t *AL, __armas_dense_t *AR,
                     const __armas_dense_t *A, int nb, int side)
{
  if (nb > A->cols) {
    nb = A->cols;
  }
  switch (side) {
  case ARMAS_PLEFT:
    __armas_submatrix(AL, A, 0, 0,  A->rows, nb);
    __armas_submatrix(AR, A, 0, nb, A->rows, A->cols-nb);
    break;
  case ARMAS_PRIGHT:
  default:
    __armas_submatrix(AL, A, 0, 0,  A->rows, A->cols-nb);
    __armas_submatrix(AR, A, 0, A->cols-nb,  A->rows, nb);
    break;
  }
}

/*
 * Repartition 1 by 2 blocks to 1 by 3 blocks.
 *
 * pRIGHT: AL | AR  -->  A0 | A1 A2 
 * pLEFT:  AL | AR  -->  A0 A1 | A2 
 *
 * Parameter As is left or right block of original 1x2 block.
 */
__ARMAS_INLINE
void __repartition_1x2to1x3(__armas_dense_t *AL, __armas_dense_t *A0,
                            __armas_dense_t *A1, __armas_dense_t *A2,
                            const __armas_dense_t *A, int nb, int direction)
{
  switch (direction) {
  case ARMAS_PLEFT:
    if (nb > AL->cols) {
      nb = AL->cols;
    }
    __armas_submatrix(A0, A, 0, 0, A->rows,  AL->cols-nb);
    __armas_submatrix(A1, A, 0, AL->cols-nb, A->rows, nb);
    __armas_submatrix(A2, A, 0, AL->cols,    A->rows, A->cols-AL->cols);
    break;
  case ARMAS_PRIGHT:
  default:
    if (AL->cols + nb > A->cols) {
      nb = A->cols - AL->cols;
    }
    __armas_submatrix(A0, A, 0, 0, A->rows, AL->cols);
    __armas_submatrix(A1, A, 0, AL->cols,    A->rows, nb);
    __armas_submatrix(A2, A, 0, AL->cols+nb, A->rows, A->cols-AL->cols-nb);
    break;
  }
}

/*
 * Repartition 1 by 2 blocks to 1 by 3 blocks.
 *
 * pRIGHT: AL | AR  --  A0 A1 | A2 
 * pLEFT:  AL | AR  <--  A0 | A1 A2 
 *
 */
__ARMAS_INLINE
void __continue_1x3to1x2(__armas_dense_t *AL, __armas_dense_t *AR,
                         __armas_dense_t *A0, __armas_dense_t *A1,
                         const __armas_dense_t *A, int direction) 
{
  int nl;
  switch (direction) {
  case ARMAS_PLEFT:
    __armas_submatrix(AL, A, 0, 0, A->rows, A0->cols);
    __armas_submatrix(AR, A, 0, A0->cols, A->rows, A->cols-A0->cols);
    break;
  case ARMAS_PRIGHT:
    nl = A0->cols + A1->cols;
    __armas_submatrix(AL, A, 0, 0, A->rows, nl);
    __armas_submatrix(AR, A, 0, nl, A->rows, A->cols-nl);
    break;
  }
}

/*
 * Partition A to 2 by 2 blocks.
 *
 *           ATL | ATR
 *  A  -->   =========
 *           ABL | ABR
 *
 * Parameter nb is initial block size for ATL in column direction and mb in row direction.
 * ATR and ABL may be nil pointers.
 */
__ARMAS_INLINE
void __partition_2x2(__armas_dense_t *ATL, __armas_dense_t *ATR,
                     __armas_dense_t *ABL, __armas_dense_t *ABR,
                     const __armas_dense_t *A, int mb, int nb, int side)
{
  switch (side) {
  case ARMAS_PTOPLEFT:
    __armas_submatrix_unsafe(ATL, A, 0, 0, mb, nb);
    if (ATR)
      __armas_submatrix(ATR, A, 0, nb, mb, A->cols-nb);
    if (ABL)
      __armas_submatrix(ABL, A, mb, 0, A->rows-mb, nb);
    __armas_submatrix_unsafe(ABR, A, mb, nb, A->rows-mb, A->cols-nb);
    break;
  case ARMAS_PBOTTOMRIGHT:
  default:
    __armas_submatrix(ATL, A, 0, 0, A->rows-mb, A->cols-nb);
    if (ATR)
      __armas_submatrix(ATR, A, 0, A->cols-nb, A->rows-mb, nb);
    if (ABL)
      __armas_submatrix(ABL, A, A->rows-mb, 0, mb, A->cols-nb);
    __armas_submatrix_unsafe(ABR, A, A->rows-mb, A->cols-nb, mb, nb);
    break;
  }
}

/*
 * Repartition 2 by 2 blocks to 3 by 3 blocks.
 *
 *                      A00 | A01 : A02
 *   ATL | ATR   nb     ===============
 *   =========   -->    A10 | A11 : A12
 *   ABL | ABR          ---------------
 *                      A20 | A21 : A22
 *
 * ATR, ABL, ABR implicitely defined by ATL and A.
 * It is valid to have either the strictly upper or lower submatrices as nil values.
 */
__ARMAS_INLINE
void __repartition_2x2to3x3(__armas_dense_t *ATL,
                            __armas_dense_t *A00, __armas_dense_t *A01, __armas_dense_t *A02,
                            __armas_dense_t *A10, __armas_dense_t *A11, __armas_dense_t *A12,
                            __armas_dense_t *A20, __armas_dense_t *A21, __armas_dense_t *A22,
                            const __armas_dense_t *A, int nb, int direction)
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
    __armas_submatrix(A00, A, 0, 0, kr, kc);
    if (A01)
      __armas_submatrix(A01, A, 0, kc, kr, nb);
    if (A02)
      __armas_submatrix(A02, A, 0, kc+nb, kr, A->cols-kc-nb);
    if (A10)
      __armas_submatrix(A10, A, kr, 0, nb, kc);
    __armas_submatrix(A11, A, kr, kc, nb, nb);
    if (A12)
      __armas_submatrix(A12, A, kr, kc+nb, nb, A->cols-kc-nb);
    if (A20)
      __armas_submatrix(A20, A, kr+nb, 0, A->rows-kr-nb, kc);
    if (A21)
      __armas_submatrix(A21, A, kr+nb, kc, A->rows-kr-nb, nb);
    __armas_submatrix(A22, A, kr+nb, kc+nb, -1, -1);
    break;
  case ARMAS_PTOPLEFT:
  default:
    // move towards top left corner
    if (nb > kc)
      nb = kc;
    if (nb > kr)
      nb = kr;
    __armas_submatrix(A00, A, 0, 0, kr-nb, kc-nb);
    if (A01)
      __armas_submatrix(A01, A, 0, kc-nb, kr-nb, nb);
    if (A02)
      __armas_submatrix(A02, A, 0, kc, kr-nb, A->cols-kc);
    if (A10)
      __armas_submatrix(A10, A, kr-nb, 0, nb, kc-nb);
    __armas_submatrix(A11, A, kr-nb, kc-nb, nb, nb);
    if (A12)
      __armas_submatrix(A12, A, kr-nb, kc, nb, A->cols-kc);
    if (A20)
      __armas_submatrix(A20, A, kr, 0, A->rows-kr, kc-nb);
    if (A21)
      __armas_submatrix(A21, A, kr, kc-nb, A->rows-kr, nb);
    __armas_submatrix(A22, A, kr, kc, -1, -1);
    break;
  }
}

/*
 * Redefine 2 by 2 blocks from 3 by 3 partition.
 *
 *                      A00 : A01 | A02
 *   ATL | ATR   nb     ---------------
 *   =========   <--    A10 : A11 | A12
 *   ABL | ABR          ===============
 *                      A20 : A21 | A22
 *
 * New division of ATL, ATR, ABL, ABR defined by diagonal entries A00, A11, A22
 */
__ARMAS_INLINE
void __continue_3x3to2x2(__armas_dense_t *ATL, __armas_dense_t *ATR,
                         __armas_dense_t *ABL, __armas_dense_t *ABR,
                         __armas_dense_t *A00, __armas_dense_t *A11, __armas_dense_t *A22,
                         const __armas_dense_t *A, int direction)
{
  //int nk = A00->rows;
  int mb = A11->cols;
  int kr = A00->rows;
  int kc = A00->cols;
  switch (direction) {
  case ARMAS_PBOTTOMRIGHT:
    __armas_submatrix(ATL, A, 0, 0,     kr+mb, kc+mb);
    if (ATR)
      __armas_submatrix(ATR, A, 0, kc+mb, kr+mb, A->cols-kc-mb);
    if (ABL)
      __armas_submatrix(ABL, A, kr+mb, 0, A->rows-kr-mb, kc+mb);
    __armas_submatrix(ABR, A, kr+mb, kc+mb, -1, -1);
    break;
  case ARMAS_PTOPLEFT:
  default:
    __armas_submatrix(ATL, A, 0, 0,  kr, kc);
    if (ATR)
      __armas_submatrix(ATR, A, 0, kc, kr, A->cols-kc);
    if (ABL)
      __armas_submatrix(ABL, A, kr, 0, A->rows-kr, A->cols-kc);
    __armas_submatrix(ABR, A, kr, kc, -1, -1);
    break;
  }
}

/*
 * Merge 1 by 1 block from 2 by 1 block.
 * 
 *          AT  
 * ABKL <-- --  
 *          AB  
 */
__ARMAS_INLINE
void __merge2x1(__armas_dense_t *ABLK, __armas_dense_t *AT, __armas_dense_t *AB)
{
  if (__armas_size(AT) == 0 && __armas_size(AB) == 0) {
    ABLK->rows = 0; ABLK->cols = 0;
    return;
  }
  if (__armas_size(AT) == 0) {
    __armas_submatrix(ABLK, AB, 0, 0, AB->rows, AB->cols);
  } else if (__armas_size(AB) == 0) {
    __armas_submatrix(ABLK, AT, 0, 0, AT->rows, AT->cols);
  } else {
    __armas_submatrix(ABLK, AT, 0, 0, AT->rows+AB->rows, AT->cols);
  }
}

/*
 * Merge 1 by 1 block from 1 by 2 block. 
 * 
 * ABLK <--  AL | AR  
 */
__ARMAS_INLINE
void __merge1x2(__armas_dense_t *ABLK, __armas_dense_t *AL, __armas_dense_t *AR)
{
  if (__armas_size(AL) == 0 && __armas_size(AR) == 0) {
    ABLK->rows = 0; ABLK->cols = 0;
    return;
  }
  if (__armas_size(AL) == 0) {
    __armas_submatrix(ABLK, AR, 0, 0, AR->rows, AR->cols);
  } else if (__armas_size(AR) == 0) {
    __armas_submatrix(ABLK, AL, 0, 0, AL->rows, AL->cols);
  } else {
    __armas_submatrix(ABLK, AL, 0, 0, AL->rows, AL->cols+AR->cols);
  }
}

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
