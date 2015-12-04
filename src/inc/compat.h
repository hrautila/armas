
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
#define __axpy  saxpy_
#define __copy  scopy_
#define __dot   sdot_
#define __nrm2  snrm2_
#define __swap  sswap_
#define __scal  sscal_
#define __asum  sasum_
#define __iamax isamax_
#define __rot   srot_
#define __rotg  srotg_
// blas2
#define __gemv  sgemv_
#define __ger   sger_
#define __symv  ssymv_
#define __syr2  ssyr2_
#define __syr   ssyr_
#define __trmv  strmv_
#define __trsv  strsv_
// blas3
#define __gemm  sgemm_
#define __symm  ssymm_
#define __syr2k ssyrk2_
#define __syrk  ssyrk_
#define __trmm  strmm_
#define __trsm  strsm_
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
#define __axpy  daxpy_
#define __copy  dcopy_
#define __dot   ddot_
#define __nrm2  dnrm2_
#define __swap  dswap_
#define __scal  dscal_
#define __asum  dasum_
#define __iamax idamax_
#define __rot   drot_
#define __rotg  drotg_
// blas2
#define __gemv  dgemv_
#define __ger   dger_
#define __symv  dsymv_
#define __syr2  dsyr2_
#define __syr   dsyr_
#define __trmv  dtrmv_
#define __trsv  dtrsv_
// blas3
#define __gemm  dgemm_
#define __symm  dsymm_
#define __syr2k dsyrk2_
#define __syrk  dsyrk_
#define __trmm  dtrmm_
#define __trsm  dtrsm_
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
