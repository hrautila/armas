
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_BASE_H
#define ARMAS_NAMES_BASE_H

#include <complex.h>

#if 0 /* untested */
/* ---------------------------------------------------------------------------
 */
typedef float complex DTYPE;
typedef float ABSTYPE;

#define ABSZERO 0.0
#define ABSONE  1.0
#define ZERO    0.0+0.0i
#define ONE     1.0+0.0i
#define TWO     2.0+0.0i
#define HALF    0.5+0.0i

#define __DATA_FORMAT "%8.1e"
#define PRINTABLE(a) (a)

// functions from math.h
#define ABS  fabsf
#define SQRT sqrtf
#define HYPOT hypotf
#define SIGN  signbitf
#define COPYSIGN copysignf
#define POW   powf
#define EXP   expf

// ------------------------------------------------------------------------------
// public matrix types
#define armas_dense_t       armas_c_dense_t
#define armas_dense         armas_c_dense

// nil matrix
#define __nil                 ((armas_dense_t *)0)

// ------------------------------------------------------------------------------
// public matrix functions
#define armas_size          armas_c_size
#define armas_init          armas_c_init
#define armas_release       armas_c_release
#define armas_alloc         armas_c_alloc
#define armas_free          armas_c_free
#define armas_make          armas_c_make
#define armas_zero          armas_c_zero
#define armas_column        armas_c_column
#define armas_row           armas_c_row
#define armas_submatrix     armas_c_submatrix
#define armas_subvector     armas_c_subvector
#define armas_submatrix_ext armas_c_submatrix_ext
#define armas_diag          armas_c_diag
#define armas_get           armas_c_get
#define armas_set           armas_c_set
#define armas_set_at        armas_c_set_at
#define armas_get_at        armas_c_get_at
#define armas_index         armas_c_index
#define armas_data          armas_c_data
#define armas_col_as_row    armas_c_col_as_row
#define armas_printf        armas_c_printf
#define armas_print         armas_c_print
#define armas_set_values    armas_c_set_values
#define armas_newcopy       armas_c_newcopy
#define armas_mcopy         armas_c_mcopy
#define armas_mscale        armas_c_mscale
#define armas_madd          armas_c_madd
#define armas_mnorm         armas_c_mnorm
#define armas_norm          armas_c_norm
#define armas_transpose     armas_c_transpose
#define armas_allclose      armas_c_allclose
#define armas_intolerance   armas_c_intolerance
#define armas_make_trm      armas_c_make_trm
#define armas_mplus         armas_c_mplus
#define armas_isvector      armas_c_isvector
// element-wise functions
#define armas_add_elems     armas_c_add_elems
#define armas_sub_elems     armas_c_sub_elems
#define armas_mul_elems     armas_c_mul_elems
#define armas_div_elems     armas_c_div_elems
#define armas_apply         armas_c_apply
#define armas_apply2        armas_c_apply2
// I/O functions
#define armas_mmload	      armas_c_mmload
#define armas_mmdump	      armas_c_mmdump
#define armas_json_dump     armas_c_json_dump
#define armas_json_load     armas_c_json_load
#define armas_json_read     armas_c_json_read
#define armas_json_write    armas_c_json_write

#define armas_submatrix_unsafe  armas_c_submatrix_unsafe
#define armas_subvector_unsafe  armas_c_subvector_unsafe
#define armas_diag_unsafe       armas_c_diag_unsafe
#define armas_get_unsafe        armas_c_get_unsafe
#define armas_set_unsafe        armas_c_set_unsafe
#define armas_get_at_unsafe     armas_c_get_at_unsafe
#define armas_set_at_unsafe     armas_c_set_at_unsafe
#define armas_row_unsafe        armas_c_row_unsafe
#define armas_column_unsafe     armas_c_column_unsafe
#define armas_index_unsafe      armas_c_index_unsafe

#define armas_index_valid  armas_c_index_valid
#define armas_real_index   armas_c_real_index


// internal matrix block routines
#define mat_partition_2x1         mat_c_partition_2x1
#define mat_repartition_2x1to3x1  mat_c_repartition_2x1to3x1
#define mat_continue_3x1to2x1     mat_c_continue_3x1to2x1
#define mat_partition_1x2         mat_c_partition_1x2
#define mat_repartition_1x2to1x3  mat_c_repartition_1x2to1x3
#define mat_continue_1x3to1x2     mat_c_continue_1x3to1x2
#define mat_partition_2x2         mat_c_partition_2x2
#define mat_repartition_2x2to3x3  mat_c_repartition_2x2to3x3
#define mat_continue_3x3to2x2     mat_c_continue_2x3to2x2
#define mat_merge2x1              mat_c_merge2x1
#define mat_merge1x2              mat_c_merge1x2

#endif  /* 0 */
#endif  /* NAMES_BASE_H */
#endif  /* DOXYGEN */
