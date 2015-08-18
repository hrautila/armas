
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_scale_plus) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"

static inline
void __scale_plus_notrans(DTYPE * __restrict dst, int ldD,
                          const DTYPE * __restrict src, int ldS, int nR, int nC,
                          const DTYPE alpha, const DTYPE beta)
{
  register DTYPE *Dc, *Dr;
  register const DTYPE *Sc, *Sr;
  register int j, i;
  register int zero;

  zero = alpha == 0.0;

  Dc = dst; Sc = src;
  for (j = 0; j < nC; j++) {
    Dr = Dc;
    Sr = Sc;
    // incrementing Dr with ldD follows the dst row
    // and incrementing Sr with one follows the column
    for (i = 0; i < nR-1; i += 2) {
      Dr[0] = zero ? beta * Sr[0] : alpha*Dr[0] + beta * Sr[0];
      Dr[1] = zero ? beta * Sr[1] : alpha*Dr[1] + beta * Sr[1];
      Dr += 2;
      Sr += 2;
    }
    if (i < nR) {
      Dr[0] = zero ? beta * Sr[0] : alpha*Dr[0] + beta * Sr[0];
    }
    Dc += ldD;
    Sc += ldS;
  }
}


// dst = beta*dst + alpha*src.T
static inline
void __scale_plus_trans(DTYPE * __restrict dst, int ldD,
                        const DTYPE * __restrict src, int ldS, int nR, int nC,
                        const DTYPE alpha, const DTYPE beta)
{
  register DTYPE *Dc, *Dr;
  register const DTYPE *Sc, *Sr;
  register int j, i;
  register int zero;

  zero = alpha == 0.0;

  Dc = dst; Sc = src;
  for (j = 0; j < nC; j++) {
    Dr = Dc;
    Sr = Sc;
    // incrementing Dr with ldD follows the dst row
    // and incrementing Sr with one follows the column
    for (i = 0; i < nR-1; i += 2) {
      Dr[0]   = zero ? beta * Sr[0] : alpha*Dr[0]   + beta * Sr[0];
      Dr[ldD] = zero ? beta * Sr[1] : alpha*Dr[ldD] + beta * Sr[1];
      Dr += ldD << 1;
      Sr += 2;
    }
    if (i < nR) {
      Dr[0] = zero ? beta * Sr[0] : alpha*Dr[0] + beta * Sr[0];
    }
    // moves Dc pointer to next row on dst
    Dc++;
    // moves Sc pointer to next column on src
    Sc += ldS;
  }
}



/*
 * A = alpha*A + beta*B
 * A = alpha*A + beta*B.T
 */
static
void __scale_plus(mdata_t *A, const mdata_t *B,
                  DTYPE alpha, DTYPE beta, int flags,
                  int S, int L, int R, int E)
{
  DTYPE *Ac;
  const DTYPE *Bc;

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  Ac = &A->md[S*A->step + R];
  if (flags & (ARMAS_TRANSB|ARMAS_TRANS)) {
    Bc = &B->md[R*B->step + S];
    __scale_plus_trans(Ac, A->step, Bc, B->step, L-S, E-R, alpha, beta);
  } else {
    Bc = &B->md[S*B->step + R];
    __scale_plus_notrans(Ac, A->step, Bc, B->step, E-R, L-S, alpha, beta);
  }
}


// A := alpha*A + beta*op(B)
int __armas_scale_plus(__armas_dense_t *A, const __armas_dense_t *B,
                       DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf)
{
  int ok;
  mdata_t *_A;
  const mdata_t *_B;

  if (!conf)
    conf = armas_conf_default();

  if (__armas_size(A) == 0 || __armas_size(B) == 0)
    return 0;

  // check consistency
  switch (flags & (ARMAS_TRANSB|ARMAS_TRANS)) {
  case ARMAS_TRANSB:
  case ARMAS_TRANS:
    ok = A->rows == B->cols && A->cols == B->rows;
    break;
  default:
    ok = A->rows == B->rows && A->cols == B->cols;
    break;
  }
  if (! ok) {
    conf->error = ARMAS_ESIZE;
    return -1;
  }

  _A = (mdata_t *)A;
  _B = (const mdata_t *)B;

  // if only one thread, just do it
  __scale_plus(_A, _B, alpha, beta, flags, 0, A->cols, 0, A->rows);
  return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
