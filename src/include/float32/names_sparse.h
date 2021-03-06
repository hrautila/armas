
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_NAMES_SPARSE_H
#define ARMAS_NAMES_SPARSE_H 1

#ifndef CONFIG_NOTYPENAMES
/* sparse.h */
#define armas_sparse_t        armas_d_sparse_t
#define armas_spvec_t         armas_d_spvec_t
#define armas_accum_t         armas_d_accum_t
#define armassp_precond_t     armassp_s_precond_t

#define armassp_bytes_needed  armassp_s_bytes_needed
#define armassp_bytes_for     armassp_s_bytes_for
#define armassp_nbytes        armassp_s_nbytes
#define armassp_index         armassp_s_index
#define armassp_index_safe    armassp_s_index_safe
#define armassp_len           armassp_s_len
#define armassp_at            armassp_s_at
#define armassp_nz            armassp_s_nz
#define armassp_iptr          armassp_s_iptr
#define armassp_data          armassp_s_data
#define armassp_size          armassp_s_size
#define armassp_value         armassp_s_value
#define armassp_nvertex       armassp_s_nvertex
#define armassp_vector        armassp_s_vector
#define armassp_clear         armassp_s_clear
#define armassp_free          armassp_s_free
#define armassp_release       armassp_s_release
/* sparse matrix core functions defined */
#define armassp_core            1
/* accumulator functions */
#define armas_accum_t         armas_d_accum_t
#define armas_accum_bytes     armas_d_accum_bytes
#define armas_accum_dim       armas_d_accum_dim
#define armas_accum_make      armas_d_accum_make
#define armas_accum_allocate  armas_d_accum_allocate
#define armas_accum_release   armas_d_accum_release
#define armas_accum_addpos    armas_d_accum_addpod
#define armas_accum_scatter   armas_d_accum_scatter
#define armas_accum_gather    armas_d_accum_gather
#define armas_accum_clear     armas_d_accum_clear
#define armas_accum_dot       armas_d_accum_dot
#define armas_accum_need      armas_d_accum_need
#define armas_accumulator     1

/* convert.c */
#define armassp_convert_to   armassp_s_convert_to
#define armassp_convert      armassp_s_convert
#define armassp_transpose_to armassp_s_transpose_to
#define armassp_transpose    armassp_s_transpose

/* sort.c */
#define armassp_sort         armassp_s_sort
#define armassp_sort_to      armassp_s_sort_to

/* copy.c */
#define armassp_copy_to      armassp_s_copy_to
#define armassp_mkcopy       armassp_s_mkcopy

/* util.c */
#define armassp_make         armassp_s_make
#define armassp_init         armassp_s_init
#define armassp_new          armassp_s_new
#define armassp_resize       armassp_s_resize
#define armassp_append       armassp_s_append
#define armassp_hasdiag      armassp_s_hasdiag

/* io.c */
#define armassp_mmload       armassp_s_mmload
#define armassp_mmdump       armassp_s_mmdump
#define armassp_pprintf      armassp_s_pprintf
#define armassp_iprintf      armassp_s_iprintf

/* trsv.c */
#define armassp_mvsolve_trm  armassp_s_mvsolve_trm
/* trmv.c */
#define armassp_mvmult_trm   armassp_s_mvmult_trm
/* symv.c */
#define armassp_mvmult_sym   armassp_s_mvmult_sym
/* gemv.c */
#define armassp_mvmult       armassp_s_mvmult
/* sparse blas core functions  */
#define armassp_blas           1

/* cgrad.c */
#define armassp_cgrad        armassp_s_cgrad
#define armassp_cgrad_w      armassp_s_cgrad_w
/* cgnr.c */
#define armassp_cgnr         armassp_s_cgnr
#define armassp_cgnr_w       armassp_s_cgnr_w
/* cgne.c */
#define armassp_cgne         armassp_s_cgne
#define armassp_cgne_w       armassp_s_cgne_w
/* pcgrad.c */
#define armassp_pcgrad       armassp_s_pcgrad
#define armassp_pcgrad_w     armassp_s_pcgrad_w

/* illtz.c */
#define armassp_init_icholz  armassp_s_init_icholz
#define armassp_icholz       armassp_s_icholz

/* iluz.c */
#define armassp_init_iluz    armassp_s_init_iluz
#define armassp_iluz         armassp_s_iluz

/* gmres.c */
#define armassp_gmres        armassp_s_gmres
#define armassp_gmres_w      armassp_s_gmres_w
/* pgmres.c */
#define armassp_pgmres       armassp_s_pgmres
#define armassp_pgmres_w     armassp_s_pgmres_w

/* diag.c */
#define armassp_mult_diag    armassp_s_mult_diag
#define armassp_add_diag     armassp_s_add_diag

/* dense.c */
#define armassp_todense      armassp_s_todense

/* add.c */
#define armassp_addto_w      armassp_s_addto_w
#define armassp_add          armassp_s_add

/* mult.c */
#define armassp_multto_w     armassp_s_multto_w
#define armassp_mult         armassp_s_mult
#define armassp_mult_nnz     armassp_s_mult_nnz

#endif /* CONFIG_NOTYPENAMES */
#endif /* ARMAS_NAMES_SPARSE_H */
