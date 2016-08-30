
// Copyright (c) Harri Rautila, 2014-2015

// This file is part of github.com/hrautila/armas package. It is free software,
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


#if defined(COMPAT)

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
// blas1
#define __axpyf  saxpy_
#define __copyf  scopy_
#define __dotf   sdot_
#define __nrm2f  snrm2_
#define __swapf  sswap_
#define __scalf  sscal_
#define __asumf  sasum_
#define __iamaxf isamax_
#define __rotf   srot_
#define __rotgf  srotg_
// blas2
#define __gemvf  sgemv_
#define __gerf   sger_
#define __symvf  ssymv_
#define __syr2f  ssyr2_
#define __syrf   ssyr_
#define __trmvf  strmv_
#define __trsvf  strsv_
// blas3
#define __gemmf  sgemm_
#define __symmf  ssymm_
#define __syr2kf ssyrk2_
#define __syrkf  ssyrk_
#define __trmmf  strmm_
#define __trsmf  strsm_
// lapack

#define __cblas_trmm   cblas_strmm
#define __cblas_trsm   cblas_strsm
#define __cblas_gemm   cblas_sgemm
#define __cblas_symm   cblas_ssymm


#else  // default is double precision float (FLOAT64)
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
// blas1
#define __axpyf  daxpy_
#define __copyf  dcopy_
#define __dotf   ddot_
#define __nrm2f  dnrm2_
#define __swapf  dswap_
#define __scalf  dscal_
#define __asumf  dasum_
#define __iamaxf idamax_
#define __rotf   drot_
#define __rotgf  drotg_
// blas2
#define __gemvf  dgemv_
#define __gerf   dger_
#define __symvf  dsymv_
#define __syr2f  dsyr2_
#define __syrf   dsyr_
#define __trmvf  dtrmv_
#define __trsvf  dtrsv_
// blas3
#define __gemmf  dgemm_
#define __symmf  dsymm_
#define __syr2kf dsyrk2_
#define __syrkf  dsyrk_
#define __trmmf  dtrmm_
#define __trsmf  dtrsm_
// lapack


#define __cblas_trmm   cblas_dtrmm
#define __cblas_trsm   cblas_dtrsm
#define __cblas_gemm   cblas_dgemm
#define __cblas_symm   cblas_dsymm

#endif  /* FLOAT64 */

#endif  /* defined(COMPAT) */


#endif  /* ARMAS_COMPAT_H */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
