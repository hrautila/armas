
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_COMPAT_H
#define ARMAS_NAMES_COMPAT_H

// blas1
#define blas_axpyf  daxpy_
#define blas_copyf  dcopy_
#define blas_dotf   ddot_
#define blas_nrm2f  dnrm2_
#define blas_swapf  dswap_
#define blas_scalf  dscal_
#define blas_asumf  dasum_
#define blas_iamaxf idamax_
#define blas_rotf   drot_
#define blas_rotgf  drotg_
// blas2
#define blas_gemvf  dgemv_
#define blas_gerf   dger_
#define blas_symvf  dsymv_
#define blas_syr2f  dsyr2_
#define blas_syrf   dsyr_
#define blas_trmvf  dtrmv_
#define blas_trsvf  dtrsv_
// blas3
#define blas_gemmf  dgemm_
#define blas_symmf  dsymm_
#define blas_syr2kf dsyrk2_
#define blas_syrkf  dsyrk_
#define blas_trmmf  dtrmm_
#define blas_trsmf  dtrsm_
// lapack


#define cblas_trmm   cblas_dtrmm
#define cblas_trsm   cblas_dtrsm
#define cblas_gemm   cblas_dgemm
#define cblas_symm   cblas_dsymm

#endif  /* NAMES_COMPAT_H */

#endif /* DOXYGEN */
