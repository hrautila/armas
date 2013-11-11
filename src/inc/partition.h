

#ifndef __ARMAS_PARTITION_H
#define __ARMAS_PARTITION_H 1


#ifndef __INLINE
#define __INLINE extern inline
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
__INLINE
void __partition_2x1(__armas_dense_t *AT, __armas_dense_t *AB,
                     __armas_dense_t *A, int nb, int side)
{
  if (nb > A->rows)
    nb = A->rows;

  switch (side) {
  case ARMAS_PTOP:
    __armas_submatrix(AT, A, 0,  0, nb, A->cols);
    __armas_submatrix(AB, A, nb, 0, A->rows-nb, A->cols);
    break;
  case ARMAS_PBOTTOM:
    __armas_submatrix(AT, A, 0, 0, A->rows-nb,  A->cols);
    __armas_submatrix(AB, A, A->rows-nb, 0, nb, A->cols);
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
__INLINE
void repartition_2x1to3x1(__armas_dense_t *AT, __armas_dense_t *A0,
                          __armas_dense_t *A1, __armas_dense_t *A2,
                          __armas_dense_t *A, int nb, int direction)
{
  switch (direction) {
  case ARMAS_PBOTTOM:
    if (nb + AT->rows > A->rows) {
      nb = A->rows - AT->rows;
    }
    __armas_submatrix(A0, A, 0, 0, AT->rows, A->cols);
    __armas_submatrix(A1, A, AT->rows, 0, nb, A->cols);
    __armas_submatrix(A2, A, AT->rows+nb, 0, A->rows-AT->rows-nb, A->cols);
    break;
  case ARMAS_PTOP:
    if (AT->rows < nb) {
      nb = AT->rows;
    }
    __armas_submatrix(A0, A, 0, 0, AT->rows-nb,  A->cols);
    __armas_submatrix(A1, A, AT->rows-nb, 0, nb, A->cols);
    __armas_submatrix(A2, A, AT->rows, 0, A->rows-AT->rows, A->cols);
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
__INLINE
void __continue_3x1to2x1(__armas_dense_t *AT, __armas_dense_t *AB,
                         __armas_dense_t *A0, __armas_dense_t *A1,
                         __armas_dense_t *A, int direction)
{
  int nr0 = A0->rows;
  int nr1 = A1->rows;
  switch (direction) {
  case ARMAS_PBOTTOM:
    __armas_submatrix(AT, A, 0,       0, nr0+nr1, A->cols);
    __armas_submatrix(AB, A, nr0+nr1, 0, A->rows-nr0-nr1, A->cols);
    break;
  case ARMAS_PTOP:
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
__INLINE
void __partition_1x2(__armas_dense_t *AL, __armas_dense_t *AR,
                     __armas_dense_t *A, int nb, int side)
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
__INLINE
void __repartition_1x2to1x3(__armas_dense_t *AL, __armas_dense_t *A0,
                            __armas_dense_t *A1, __armas_dense_t *A2,
                            __armas_dense_t *A, int nb, int direction)
{
  switch (direction) {
  case ARMAS_PLEFT:
    if (AL->cols + nb > A->cols) {
      nb = A->cols - AL->cols;
    }
    __armas_submatrix(A0, A, 0, 0, A->rows, AL->cols);
    __armas_submatrix(A1, A, 0, AL->cols,    A->rows, nb);
    __armas_submatrix(A2, A, 0, AL->cols+nb, A->rows, A->cols-AL->cols-nb);
    break;
  case ARMAS_PRIGHT:
    if (nb > AL->cols) {
      nb = AL->cols;
    }
    __armas_submatrix(A0, A, 0, 0, A->rows,  AL->cols-nb);
    __armas_submatrix(A1, A, 0, AL->cols-nb, A->rows, nb);
    __armas_submatrix(A2, A, 0, AL->cols,    A->rows, A->cols-AL->cols);
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
__INLINE
void __continue_1x3to1x2(__armas_dense_t *AL, __armas_dense_t *AR,
                         __armas_dense_t *A0, __armas_dense_t *A1,
                         __armas_dense_t *A, int direction) 
{
  switch (direction) {
  case ARMAS_PLEFT:
    __armas_submatrix(AL, A, 0, 0, A->rows, A0->cols+A1->cols);
    __armas_submatrix(AR, A, 0, AL->cols, A->rows, A->cols-AL->cols);
    break;
  case ARMAS_PRIGHT:
    __armas_submatrix(AL, A, 0, 0, A->rows, A0->cols);
    __armas_submatrix(AR, A, 0, A0->cols, A->rows, A->cols-A0->cols);
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
__INLINE
void __partition_2x2(__armas_dense_t *ATL, __armas_dense_t *ATR,
                     __armas_dense_t *ABL, __armas_dense_t *ABR,
                     __armas_dense_t *A, int mb, int nb, int side)
{
  switch (side) {
  case ARMAS_PTOPLEFT:
    __armas_submatrix(ATL, A, 0, 0, mb, nb);
    if (ATR)
      __armas_submatrix(ATR, A, 0, nb, mb, A->cols-nb);
    if (ABL)
      __armas_submatrix(ABL, A, mb, 0, A->rows-mb, nb);
    __armas_submatrix(ABR, A, mb, nb, -1, -1);
    break;
  case ARMAS_PBOTTOMRIGHT:
    __armas_submatrix(ATL, A, 0, 0, A->rows-mb, A->cols-nb);
    if (ATR)
      __armas_submatrix(ATR, A, 0, A->cols-nb, A->rows-mb, nb);
    if (ABL)
      __armas_submatrix(ABL, A, A->rows-mb, 0, mb, nb);
    __armas_submatrix(ABR, A, A->rows-mb, A->cols-nb, -1, -1);
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
void __repartition_2x2to3x3(__armas_dense_t *ATL,
                            __armas_dense_t *A00, __armas_dense_t *A01, __armas_dense_t *A02,
                            __armas_dense_t *A10, __armas_dense_t *A11, __armas_dense_t *A12,
                            __armas_dense_t *A20, __armas_dense_t *A21, __armas_dense_t *A22,
                            __armas_dense_t *A, int nb, int direction)
{
  int nk = ATL->rows;
  switch (direction) {
  case ARMAS_PBOTTOMRIGHT:
    if (nk + nb > A->cols) {
      nb = A->cols - nk;
    }
    __armas_submatrix(A00, A, 0, 0, nk, nk);
    if (A01)
      __armas_submatrix(A01, A, 0, nk, nk, nb);
    if (A02)
      __armas_submatrix(A02, A, 0, nk+nb, nk, A->cols-nk-nb);
    if (A10)
      __armas_submatrix(A10, A, nk, 0, nb, nk);
    __armas_submatrix(A11, A, nk, nk, nb, nb);
    if (A12)
      __armas_submatrix(A12, A, nk, nk+nb, nb, A->cols-nk-nb);
    if (A20)
      __armas_submatrix(A20, A, nk+nb, 0, A->rows-nk-nb, nk);
    if (A21)
      __armas_submatrix(A21, A, nk+nb, nk, A->rows-nk-nb, nb);
    __armas_submatrix(A22, A, nk+nb, nk+nb, -1, -1);
    break;
  case ARMAS_PTOPLEFT:
    // move towards top left corner
    if (nb > nk)
      nb = nk;
    __armas_submatrix(A00, A, 0, 0, nk-nb, nk-nb);
    if (A01)
      __armas_submatrix(A01, A, 0, nk-nb, nk-nb, nb);
    if (A02)
      __armas_submatrix(A02, A, 0, nk, nk-nb, A->cols-nk);
    if (A10)
      __armas_submatrix(A10, A, nk-nb, 0, nb, nk-nb);
    __armas_submatrix(A11, A, nk-nb, nk-nb, nb, nb);
    if (A12)
      __armas_submatrix(A12, A, nk-nb, nk, nb, A->cols-nk);
    if (A20)
      __armas_submatrix(A20, A, nk, 0, A->rows-nk, nk-nb);
    if (A21)
      __armas_submatrix(A21, A, nk, nk-nb, A->rows-nk, nb);
    __armas_submatrix(A22, A, nk, nk, -1, -1);
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
__INLINE
void __continue_3x3to2x2(__armas_dense_t *ATL, __armas_dense_t *ATR,
                         __armas_dense_t *ABL, __armas_dense_t *ABR,
                         __armas_dense_t *A00, __armas_dense_t *A11, __armas_dense_t *A22,
                         __armas_dense_t *A, int direction)
{
  int nk = A00->rows;
  int mb = A11->cols;

  switch (direction) {
  case ARMAS_PBOTTOMRIGHT:
    __armas_submatrix(ATL, A, 0, 0,     nk+mb, nk+mb);
    __armas_submatrix(ATR, A, 0, nk+mb, nk+mb, A->cols-nk-mb);
    
    __armas_submatrix(ABL, A, nk+mb, 0, A->rows-nk-mb, nk+mb);
    __armas_submatrix(ABR, A, nk+mb, nk+mb, -1, -1);
    break;
  case ARMAS_PTOPLEFT:
    __armas_submatrix(ATL, A, 0, 0,  nk, nk);
    __armas_submatrix(ATR, A, 0, nk, nk, A->cols-nk);

    __armas_submatrix(ABL, A, nk, 0, A->rows-nk, A->cols-nk);
    __armas_submatrix(ATL, A, nk, nk, -1, -1);
    break;
  }
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
