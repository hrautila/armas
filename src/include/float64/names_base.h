
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_BASE_H
#define ARMAS_NAMES_BASE_H

/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
typedef double DTYPE;
typedef double ABSTYPE;

#define ABSZERO (double)0.0
#define ABSONE  (double)1.0
#define ZERO    (double)0.0
#define ONE     (double)1.0
#define TWO     (double)2.0
#define HALF    0.5L

#define __DATA_FORMAT "%8.1e"
#define PRINTABLE(a) (a)T

// functions from math.h
#define ABS  fabs
#define SQRT sqrt
#define HYPOT hypot
#define SIGN  signbit
#define COPYSIGN copysign
#define POW   pow
#define EXP   exp
#define MAX   fmax
#define MIN   fmin

#define SAFEMIN     DBL_MIN
// machine accurancy as in LAPACK library
#define EPS         (DBL_EPSILON/2.0)

// nil matrix
#define __nil                 ((armas_dense_t *)0)

#ifndef CONFIG_NOTYPENAMES
// ------------------------------------------------------------------------------
// public matrix types
#define armas_dense_t       armas_d_dense_t
#define armas_dense         armas_d_dense

#define armas_valuefunc_t   armas_d_valuefunc_t
#define armas_constfunc_t   armas_d_constfunc_t
#define armas_operator_t    armas_d_operator_t
#define armas_operator2_t   armas_d_operator2_t
#define armas_iterator_t    armas_d_iterator_t

// ------------------------------------------------------------------------------
// public matrix functions
#define armas_size          armas_d_size
#define armas_init          armas_d_init
#define armas_release       armas_d_release
#define armas_alloc         armas_d_alloc
#define armas_free          armas_d_free
#define armas_make          armas_d_make
#define armas_zero          armas_d_zero
#define armas_column        armas_d_column
#define armas_row           armas_d_row
#define armas_submatrix     armas_d_submatrix
#define armas_subvector     armas_d_subvector
#define armas_submatrix_ext armas_d_submatrix_ext
#define armas_diag          armas_d_diag
#define armas_get           armas_d_get
#define armas_set           armas_d_set
#define armas_set_at        armas_d_set_at
#define armas_get_at        armas_d_get_at
#define armas_index         armas_d_index
#define armas_data          armas_d_data
#define armas_col_as_row    armas_d_col_as_row
#define armas_printf        armas_d_printf
#define armas_print         armas_d_print
#define armas_set_values    armas_d_set_values
#define armas_newcopy       armas_d_newcopy
#define armas_mcopy         armas_d_mcopy
#define armas_mscale        armas_d_mscale
#define armas_madd          armas_d_madd
#define armas_mnorm         armas_d_mnorm
#define armas_norm          armas_d_norm
#define armas_transpose     armas_d_transpose
#define armas_allclose      armas_d_allclose
#define armas_intolerance   armas_d_intolerance
#define armas_make_trm      armas_d_make_trm
#define armas_mplus         armas_d_mplus
#define armas_isvector      armas_d_isvector
// element-wise functions
#define armas_add_elems     armas_d_add_elems
#define armas_sub_elems     armas_d_sub_elems
#define armas_mul_elems     armas_d_mul_elems
#define armas_div_elems     armas_d_div_elems
#define armas_apply         armas_d_apply
#define armas_apply2        armas_d_apply2
#define armas_iterate       armas_d_iterate
// I/O functions
#define armas_mmload	    armas_d_mmload
#define armas_mmdump	    armas_d_mmdump
#define armas_json_dump     armas_d_json_dump
#define armas_json_load     armas_d_json_load
#define armas_json_read     armas_d_json_read
#define armas_json_write    armas_d_json_write

#define armas_submatrix_unsafe  armas_d_submatrix_unsafe
#define armas_subvector_unsafe  armas_d_subvector_unsafe
#define armas_diag_unsafe       armas_d_diag_unsafe
#define armas_get_unsafe        armas_d_get_unsafe
#define armas_set_unsafe        armas_d_set_unsafe
#define armas_get_at_unsafe     armas_d_get_at_unsafe
#define armas_set_at_unsafe     armas_d_set_at_unsafe
#define armas_row_unsafe        armas_d_row_unsafe
#define armas_column_unsafe     armas_d_column_unsafe
#define armas_index_unsafe      armas_d_index_unsafe

#define armas_index_valid  armas_d_index_valid
#define armas_real_index   armas_d_real_index


// internal matrix block routines
#define mat_partition_2x1         mat_d_partition_2x1
#define mat_repartition_2x1to3x1  mat_d_repartition_2x1to3x1
#define mat_continue_3x1to2x1     mat_d_continue_3x1to2x1
#define mat_partition_1x2         mat_d_partition_1x2
#define mat_repartition_1x2to1x3  mat_d_repartition_1x2to1x3
#define mat_continue_1x3to1x2     mat_d_continue_1x3to1x2
#define mat_partition_2x2         mat_d_partition_2x2
#define mat_repartition_2x2to3x3  mat_d_repartition_2x2to3x3
#define mat_continue_3x3to2x2     mat_d_continue_2x3to2x2
#define mat_merge2x1              mat_d_merge2x1
#define mat_merge1x2              mat_d_merge1x2
#define vec_partition_2x1         vec_d_partition_2x1
#define vec_repartition_2x1to3x1  vec_d_repartition_2x1to3x1
#define vec_continue_3x1to2x1     vec_d_continue_3x1to2x1

#endif /* CONFIG_NOTYPENAMES */
#endif /* NAMES_BASE_H */

#endif /* __DOXYGEN */
