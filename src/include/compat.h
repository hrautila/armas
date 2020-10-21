
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_COMPAT_H
#define __ARMAS_COMPAT_H 1

#include <math.h>
#include <complex.h>

#include "dtype.h"

enum cblas_const {
  CBLAS_ROW_MAJOR = 101,
  CBLAS_COL_MAJOR = 102,
  CBLAS_NOTRANS = 111,
  CBLAS_TRANS = 112,
  CBLAS_CONJ_TRANS = 113,
  CBLAS_UPPER = 121,
  CBLAS_LOWER = 122,
  CBLAS_NONUNIT = 131,
  CBLAS_UNIT = 132,
  CBLAS_LEFT = 141,
  CBLAS_RIGHT = 142
};


#if defined(CONFIG_COMPAT)

#define CBLAS_INDEX size_t 

// from cblas.h
enum CBLAS_ORDER 	{CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE 	{CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO		{CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG		{CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE		{CblasLeft=141, CblasRight=142};

// from lapacke.h
#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102

#define LAPACK_WORK_MEMORY_ERROR       -1010
#define LAPACK_TRANSPOSE_MEMORY_ERROR  -1011

#ifdef COMPLEX128
/* ---------------------------------------------------------------------------
 * Definitions for double precision complex numbers.
 */

#elif COMPLEX64
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */

#elif FLOAT32
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */
#include "float32/names_compat.h"

#else  // default is double precision float (FLOAT64)
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
#include "float64/names_compat.h"

#endif  /* FLOAT64 */

#endif  /* CONFIG_COMPAT */
#endif  /* ARMAS_COMPAT_H */
