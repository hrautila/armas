
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_BASE_H
#define ARMAS_NAMES_BASE_H

/* ---------------------------------------------------------------------------
 * Definitions for floating .
 */
typedef float DTYPE;
typedef float ABSTYPE;

#define ABSZERO (float)0.0
#define ABSONE  (float)1.0
#define ZERO    (float)0.0
#define ONE     (float)1.0
#define TWO     (float)2.0
#define HALF    0.5

#define __DATA_FORMAT "%8.1e"
#define PRINTABLE(a) (a)

// functions from math.h
#define ABS  fabsf
#define SQRT sqrtf
#define HYPOT hypotf
#define SIGN  signbit
#define COPYSIGN copysignf
#define POW   powf
#define EXP   expf
#define MAX   fmaxf
#define MIN   fminf

#define SAFEMIN     FLT_MIN
// machine accurancy as in LAPACK library
#define EPS         (FLT_EPSILON/2.0)

// nil matrix
#define __nil                 ((armas_dense_t *)0)

#ifndef CONFIG_NOTYPENAMES
// ------------------------------------------------------------------------------
// public matrix types
#define armas_dense_t       armas_s_dense_t
#define armas_dense         armas_s_dense


// ------------------------------------------------------------------------------
// public matrix functions
#define armas_size          armas_s_size
#define armas_init          armas_s_init
#define armas_release       armas_s_release
#define armas_alloc         armas_s_alloc
#define armas_free          armas_s_free
#define armas_make          armas_s_make
#define armas_zero          armas_s_zero
#define armas_column        armas_s_column
#define armas_row           armas_s_row
#define armas_submatrix     armas_s_submatrix
#define armas_subvector     armas_s_subvector
#define armas_submatrix_ext armas_s_submatrix_ext
#define armas_diag          armas_s_diag
#define armas_get           armas_s_get
#define armas_set           armas_s_set
#define armas_set_at        armas_s_set_at
#define armas_get_at        armas_s_get_at
#define armas_index         armas_s_index
#define armas_data          armas_s_data
#define armas_col_as_row    armas_s_col_as_row
#define armas_printf        armas_s_printf
#define armas_print         armas_s_print
#define armas_set_values    armas_s_set_values
#define armas_newcopy       armas_s_newcopy
#define armas_mcopy         armas_s_mcopy
#define armas_mscale        armas_s_mscale
#define armas_madd          armas_s_madd
#define armas_mnorm         armas_s_mnorm
#define armas_norm          armas_s_norm
#define armas_transpose     armas_s_transpose
#define armas_allclose      armas_s_allclose
#define armas_intolerance   armas_s_intolerance
#define armas_make_trm      armas_s_make_trm
#define armas_mplus         armas_s_mplus
#define armas_isvector      armas_s_isvector
// element-wise functions
#define armas_add_elems     armas_s_add_elems
#define armas_sub_elems     armas_s_sub_elems
#define armas_mul_elems     armas_s_mul_elems
#define armas_div_elems     armas_s_div_elems
#define armas_apply         armas_s_apply
#define armas_apply2        armas_s_apply2
// I/O functions
#define armas_mmload	      armas_s_mmload
#define armas_mmdump	      armas_s_mmdump
#define armas_json_dump     armas_s_json_dump
#define armas_json_load     armas_s_json_load
#define armas_json_read     armas_s_json_read
#define armas_json_write    armas_s_json_write

#define armas_submatrix_unsafe  armas_s_submatrix_unsafe
#define armas_subvector_unsafe  armas_s_subvector_unsafe
#define armas_diag_unsafe       armas_s_diag_unsafe
#define armas_get_unsafe        armas_s_get_unsafe
#define armas_set_unsafe        armas_s_set_unsafe
#define armas_get_at_unsafe     armas_s_get_at_unsafe
#define armas_set_at_unsafe     armas_s_set_at_unsafe
#define armas_row_unsafe        armas_s_row_unsafe
#define armas_column_unsafe     armas_s_column_unsafe
#define armas_index_unsafe      armas_s_index_unsafe

#define armas_index_valid  armas_s_index_valid
#define armas_real_index   armas_s_real_index


// internal matrix block routines
#define mat_partition_2x1         mat_s_partition_2x1
#define mat_repartition_2x1to3x1  mat_s_repartition_2x1to3x1
#define mat_continue_3x1to2x1     mat_s_continue_3x1to2x1
#define mat_partition_1x2         mat_s_partition_1x2
#define mat_repartition_1x2to1x3  mat_s_repartition_1x2to1x3
#define mat_continue_1x3to1x2     mat_s_continue_1x3to1x2
#define mat_partition_2x2         mat_s_partition_2x2
#define mat_repartition_2x2to3x3  mat_s_repartition_2x2to3x3
#define mat_continue_3x3to2x2     mat_s_continue_2x3to2x2
#define mat_merge2x1              mat_s_merge2x1
#define mat_merge1x2              mat_s_merge1x2

#endif /* CONFIG_NOTYPENAMES */
#endif  /* NAMES_BASE_H */

#endif /* DOXYGEN */
