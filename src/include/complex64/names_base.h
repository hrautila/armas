
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_BASE_H
#define ARMAS_NAMES_BASE_H

#include <complex.h>
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
typedef double complex DTYPE;
typedef double ABSTYPE;

#define ABSZERO 0.0L
#define ABSONE  1.0L
#define ZERO    0.0L + 0.0Li
#define ONE     1.0L + 0.0Li
#define TWO     2.0L + 0.0Li
#define HALF    0.5L + 0.0Li

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
#define armas_x_dense_t       armas_z_dense_t
#define armas_x_dense         armas_z_dense

// nil matrix
#define __nil                 ((armas_x_dense_t *)0)

// ------------------------------------------------------------------------------
// public matrix functions
#define armas_x_size          armas_z_size
#define armas_x_init          armas_z_init
#define armas_x_release       armas_z_release
#define armas_x_alloc         armas_z_alloc
#define armas_x_free          armas_z_free
#define armas_x_make          armas_z_make
#define armas_x_column        armas_z_column
#define armas_x_row           armas_z_row
#define armas_x_submatrix     armas_z_submatrix
#define armas_x_subvector     armas_z_subvector
#define armas_x_submatrix_ext armas_z_submatrix_ext
#define armas_x_diag          armas_z_diag
#define armas_x_get           armas_z_get
#define armas_x_set           armas_z_set
#define armas_x_set_at        armas_z_set_at
#define armas_x_get_at        armas_z_get_at
#define armas_x_index         armas_z_index
#define armas_x_data          armas_z_data
#define armas_x_col_as_row    armas_z_col_as_row
#define armas_x_printf        armas_z_printf
#define armas_x_print         armas_z_print
#define armas_x_set_values    armas_z_set_values
#define armas_x_newcopy       armas_z_newcopy
#define armas_x_mcopy         armas_z_mcopy
#define armas_x_mscale        armas_z_mscale
#define armas_x_madd          armas_z_madd
#define armas_x_mnorm         armas_z_mnorm
#define armas_x_norm          armas_z_norm
#define armas_x_transpose     armas_z_transpose
#define armas_x_allclose      armas_z_allclose
#define armas_x_intolerance   armas_z_intolerance
#define armas_x_make_trm      armas_z_make_trm
#define armas_x_mplus         armas_z_mplus
#define armas_x_isvector      armas_z_isvector
// element-wise functions
#define armas_x_add_elems     armas_z_add_elems
#define armas_x_sub_elems     armas_z_sub_elems
#define armas_x_mul_elems     armas_z_mul_elems
#define armas_x_div_elems     armas_z_div_elems
#define armas_x_apply         armas_z_apply
#define armas_x_apply2        armas_z_apply2
// I/O functions
#define armas_x_mmload	      armas_z_mmload
#define armas_x_mmdump	      armas_z_mmdump
#define armas_x_json_dump     armas_z_json_dump
#define armas_x_json_load     armas_z_json_load
#define armas_x_json_read     armas_z_json_read
#define armas_x_json_write    armas_z_json_write

#define armas_x_submatrix_unsafe  armas_z_submatrix_unsafe
#define armas_x_subvector_unsafe  armas_z_subvector_unsafe
#define armas_x_diag_unsafe       armas_z_diag_unsafe
#define armas_x_get_unsafe        armas_z_get_unsafe
#define armas_x_set_unsafe        armas_z_set_unsafe
#define armas_x_get_at_unsafe     armas_z_get_at_unsafe
#define armas_x_set_at_unsafe     armas_z_set_at_unsafe
#define armas_x_row_unsafe        armas_z_row_unsafe
#define armas_x_column_unsafe     armas_z_column_unsafe
#define armas_x_index_unsafe      armas_z_index_unsafe

#define armas_x_index_valid  armas_z_index_valid
#define armas_x_real_index   armas_z_real_index


// internal matrix block routines
#define mat_partition_2x1         mat_z_partition_2x1
#define mat_repartition_2x1to3x1  mat_z_repartition_2x1to3x1
#define mat_continue_3x1to2x1     mat_z_continue_3x1to2x1
#define mat_partition_1x2         mat_z_partition_1x2
#define mat_repartition_1x2to1x3  mat_z_repartition_1x2to1x3
#define mat_continue_1x3to1x2     mat_z_continue_1x3to1x2
#define mat_partition_2x2         mat_z_partition_2x2
#define mat_repartition_2x2to3x3  mat_z_repartition_2x2to3x3
#define mat_continue_3x3to2x2     mat_z_continue_2x3to2x2
#define mat_merge2x1              mat_z_merge2x1
#define mat_merge1x2              mat_z_merge1x2

#endif  /* NAMES_BASE_H */

#endif /* __DOXYGEN */
