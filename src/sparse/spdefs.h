
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_SPDEFS_H
#define __ARMAS_SPDEFS_H 1

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>

#include "dtype.h"

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

#else  // default is double precision float (FLOAT64)
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
#ifndef __EPSILON
#define __EPSILON  DBL_EPSILON
#endif

/* sparse.h */
#define armas_x_sparse_t        armas_d_sparse_t
#define armassp_x_bytes_needed  armassp_d_bytes_needed
#define armassp_x_bytes_for     armassp_d_bytes_for
#define armassp_x_nbytes        armassp_d_nbytes
#define armassp_x_index         armassp_d_index
#define armassp_x_index_safe    armassp_d_index_safe
#define armassp_x_len           armassp_d_len
#define armassp_x_at            armassp_d_at
#define armassp_x_nz            armassp_d_nz
#define armassp_x_iptr          armassp_d_iptr
#define armassp_x_data          armassp_d_data
#define armassp_x_size          armassp_d_size
#define armassp_x_value         armassp_d_value
#define armassp_x_nvertex       armassp_d_nvertex
#define armassp_x_vector        armassp_d_vector
#define armassp_x_clear         armassp_d_clear
#define armassp_x_free          armassp_d_free
#define armassp_x_release       armassp_d_release
/* sparse matrix core functions defined */
#define armassp_core            1
/* accumulator functions */
#define armas_x_accum_t         armas_d_accum_t
#define armas_x_accum_bytes     armas_d_accum_bytes
#define armas_x_accum_dim       armas_d_accum_dim
#define armas_x_accum_make      armas_d_accum_make
#define armas_x_accum_allocate  armas_d_accum_allocate
#define armas_x_accum_release   armas_d_accum_release
#define armas_x_accum_addpos    armas_d_accum_addpod
#define armas_x_accum_scatter   armas_d_accum_scatter
#define armas_x_accum_gather    armas_d_accum_gather
#define armas_x_accum_clear     armas_d_accum_clear
#define armas_x_accum_dot       armas_d_accum_dot
#define armas_x_accum_need      armas_d_accum_need
#define armas_x_accumulator     1

/* convert.c */
#define armassp_x_convert_to   armassp_d_convert_to
#define armassp_x_convert      armassp_d_convert
#define armassp_x_transpose_to armassp_d_transpose_to
#define armassp_x_transpose    armassp_d_transpose

/* sort.c */
#define armassp_x_sort         armassp_d_sort
#define armassp_x_sort_to      armassp_d_sort_to

/* copy.c */
#define armassp_x_copy_to      armassp_d_copy_to
#define armassp_x_mkcopy       armassp_d_mkcopy

/* util.c */
#define armassp_x_make         armassp_d_make
#define armassp_x_init         armassp_d_init
#define armassp_x_new          armassp_d_new
#define armassp_x_resize       armassp_d_resize
#define armassp_x_append       armassp_d_append

/* io.c */
#define armassp_x_mmload       armassp_d_mmload
#define armassp_x_mmdump       armassp_d_mmdump
#define armassp_x_pprintf      armassp_d_pprintf
#define armassp_x_iprintf      armassp_d_iprintf

/* trsv.c */
#define armassp_x_mvsolve_trm  armassp_d_mvsolve_trm
/* trmv.c */
#define armassp_x_mvmult_trm   armassp_d_mvmult_trm
/* symv.c */
#define armassp_x_mvmult_sym   armassp_d_mvmult_sym
/* gemv.c */
#define armassp_x_mvmult       armassp_d_mvmult
/* sparse blas core functions  */
#define armassp_blas           1

/* cgrad.c */
#define armassp_x_cgrad        armassp_d_cgrad
#define armassp_x_cgrad_w      armassp_d_cgrad_w
#define armassp_x_pcgrad       armassp_d_pcgrad
#define armassp_x_pcgrad_w     armassp_d_pcgrad_w

/* illtz.c */
#define armassp_x_init_icholz  armassp_d_init_icholz
#define armassp_x_icholz       armassp_d_icholz

/* gmres.c */
#define armassp_x_gmres        armassp_d_gmres
#define armassp_x_gmres_w      armassp_d_gmres_w

/* diag.c */
#define armassp_x_mult_diag    armassp_d_mult_diag
#define armassp_x_add_diag     armassp_d_add_diag

/* dense.c */
#define armassp_x_todense      armassp_d_todense

/* add.c */
#define armassp_x_addto_w      armassp_d_addto_w
#define armassp_x_add          armassp_d_add

/* mult.c */
#define armassp_x_multto_w     armassp_d_multto_w
#define armassp_x_mult         armassp_d_mult
#define armassp_x_mult_nnz     armassp_d_mult_nnz


#endif  /* FLOAT64 */



#endif  /* ARMAS_SPDEFS_H */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
