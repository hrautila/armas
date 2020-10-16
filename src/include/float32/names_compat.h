
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_COMPAT_H
#define ARMAS_NAMES_COMPAT_H

// blas1
#define blas_axpyf  saxpy_
#define blas_copyf  scopy_
#define blas_dotf   sdot_
#define blas_nrm2f  snrm2_
#define blas_swapf  sswap_
#define blas_scalf  sscal_
#define blas_asumf  sasum_
#define blas_iamaxf isamax_
#define blas_rotf   srot_
#define blas_rotgf  srotg_
// blas2
#define blas_gemvf  sgemv_
#define blas_gerf   sger_
#define blas_symvf  ssymv_
#define blas_syr2f  ssyr2_
#define blas_syrf   ssyr_
#define blas_trmvf  strmv_
#define blas_trsvf  strsv_
// blas3
#define blas_gemmf  sgemm_
#define blas_symmf  ssymm_
#define blas_syr2kf ssyrk2_
#define blas_syrkf  ssyrk_
#define blas_trmmf  strmm_
#define blas_trsmf  strsm_
// lapack


#define cblas_trmm   cblas_strmm
#define cblas_trsm   cblas_strsm
#define cblas_gemm   cblas_sgemm
#define cblas_symm   cblas_ssymm

#endif  /* NAMES_COMPAT_H */

#endif /* DOXYGEN */
