
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_LAPACK_AUXILIARY_H
#define ARMAS_LAPACK_AUXILIARY_H 1

#include "internal_lapack.h"

/*
 * \brief Compute Wilkinson shift from trailing 2-by-2 submatrix
 *
 * \verbatim
 *   ( tn1  tnn1 )
 *   ( tnn1 tn   )
 * \endverbatim
 *
 * Stable formula for the Wilkinson's shift (Hogben 2007, 42.3)
 *   d = (tn1 - tn)/2.0
 *   u = tn - sign(d)*tnn1^2/(abs(d) + sqrt(d^2 + tnn1^2))
 *
 */
static inline
DTYPE wilkinson_shift(DTYPE tn1, DTYPE tnn1, DTYPE tn)
{
    DTYPE d, tsq;
    d = (tn1 - tn) / 2.0;
    tsq = HYPOT(d, tnn1);
    return tn - COPYSIGN((tnn1 / (ABS(d) + tsq)) * tnn1, d);
}


/*
 * \brief Compute eigenvalues of 2x2 matrix.
 *
 * \param e1, e2 [out]
 *      Computed eigenvalue
 * \param a, b, c, d [in]
 *      Matrix elements
 *
 *  Characteristic function for solving eigenvalues:
 * \verbatim
 *   det A-Ix = det ( a-x   b  )
 *                  (  c   d-x )
 * \endverbatim
 *      x^2 - (a + d)*x + (a*d - c*b) = 0 =>
 *      x^2 - 2*T*x + D = 0, T = (a + d)/2, D = a*d - c*b
 *
 *      e1 = T + sqrt(T*T - D)
 *      e2 = T - sqrt(T*T - D)
 */
static inline
void eigen2x2(DTYPE * e1, DTYPE * e2, DTYPE a, DTYPE b, DTYPE c, DTYPE d)
{
    DTYPE T, D;

    T = (a + d) / 2.0;
    D = a * d - b * c;
    armas_x_qdroots(e1, e2, 1.0, T, D);
}

#endif
