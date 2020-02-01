
// Copyright (c) Harri Rautila, 2013

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
#define PRINTABLE(a) (a)

// functions from math.h
#define ABS  fabs
#define SQRT sqrt
#define HYPOT hypot
#define SIGN  signbit
#define COPYSIGN copysign
#define POW   pow
#define EXP   exp

// ------------------------------------------------------------------------------
// public matrix types
#define armas_x_dense_t       armas_d_dense_t
#define armas_x_dense         armas_d_dense

// nil matrix
#define __nil                 ((armas_x_dense_t *)0)

// ------------------------------------------------------------------------------
// public matrix functions
#define armas_x_size          armas_d_size
#define armas_x_init          armas_d_init
#define armas_x_release       armas_d_release
#define armas_x_alloc         armas_d_alloc
#define armas_x_free          armas_d_free
#define armas_x_make          armas_d_make
#define armas_x_column        armas_d_column
#define armas_x_row           armas_d_row
#define armas_x_submatrix     armas_d_submatrix
#define armas_x_subvector     armas_d_subvector
#define armas_x_submatrix_ext armas_d_submatrix_ext
#define armas_x_diag          armas_d_diag
#define armas_x_get           armas_d_get
#define armas_x_set           armas_d_set
#define armas_x_set_at        armas_d_set_at
#define armas_x_get_at        armas_d_get_at
#define armas_x_index         armas_d_index
#define armas_x_data          armas_d_data
#define armas_x_col_as_row    armas_d_col_as_row
#define armas_x_printf        armas_d_printf
#define armas_x_print         armas_d_print
#define armas_x_set_values    armas_d_set_values
#define armas_x_newcopy       armas_d_newcopy
#define armas_x_mcopy         armas_d_mcopy
#define armas_x_mscale        armas_d_mscale
#define armas_x_madd          armas_d_madd
#define armas_x_mnorm         armas_d_mnorm
#define armas_x_norm          armas_d_norm
#define armas_x_transpose     armas_d_transpose
#define armas_x_allclose      armas_d_allclose
#define armas_x_intolerance   armas_d_intolerance
#define armas_x_make_trm      armas_d_make_trm
#define armas_x_mplus         armas_d_mplus
#define armas_x_isvector      armas_d_isvector
// element-wise functions
#define armas_x_add_elems     armas_d_add_elems
#define armas_x_sub_elems     armas_d_sub_elems
#define armas_x_mul_elems     armas_d_mul_elems
#define armas_x_div_elems     armas_d_div_elems
#define armas_x_apply         armas_d_apply
#define armas_x_apply2        armas_d_apply2
// I/O functions
#define armas_x_mmload	      armas_d_mmload
#define armas_x_mmdump	      armas_d_mmdump
#define armas_x_json_dump     armas_d_json_dump
#define armas_x_json_load     armas_d_json_load
#define armas_x_json_read     armas_d_json_read
#define armas_x_json_write    armas_d_json_write

#define armas_x_submatrix_unsafe  armas_d_submatrix_unsafe
#define armas_x_subvector_unsafe  armas_d_subvector_unsafe
#define armas_x_diag_unsafe       armas_d_diag_unsafe
#define armas_x_get_unsafe        armas_d_get_unsafe
#define armas_x_set_unsafe        armas_d_set_unsafe
#define armas_x_get_at_unsafe     armas_d_get_at_unsafe
#define armas_x_set_at_unsafe     armas_d_set_at_unsafe
#define armas_x_row_unsafe        armas_d_row_unsafe
#define armas_x_column_unsafe     armas_d_column_unsafe
#define armas_x_index_unsafe      armas_d_index_unsafe

#define armas_x_index_valid  armas_d_index_valid
#define armas_x_real_index   armas_d_real_index


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

#endif  /* NAMES_BASE_H */

#endif /* __DOXYGEN */
