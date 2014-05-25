
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_INTERNAL_LAPACK_H
#define __ARMAS_INTERNAL_LAPACK_H 1

// internal declarations for lapack functionality.

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

// Bunch-Kauffman algorithm constant ALPHA = (1 + sqrt(17))/8
#define bkALPHA  0.64038820320220756872

enum pivot_dir {
  PIVOT_FORWARD = 0,
  PIVOT_BACKWARD = 1
};

#define ONERROR(exp) \
  do { if ( (exp) ) return -1; } while (0)


extern int compute_lb(int M, int N, int wsize, WSSIZE wsizer);

// helper functions
extern
void __swap_rows(__armas_dense_t *A, int src, int dst, armas_conf_t *conf);

extern
void __swap_cols(__armas_dense_t *A, int src, int dst, armas_conf_t *conf);
 
extern
void __scale_pivots(armas_pivot_t *P, int offset);

extern
void __apply_pivots(__armas_dense_t *A, armas_pivot_t *P, armas_conf_t *conf);

extern
void __apply_row_pivots(__armas_dense_t *A, armas_pivot_t *P, int dir, armas_conf_t *conf);

extern
void __apply_col_pivots(__armas_dense_t *A, armas_pivot_t *P, int dir, armas_conf_t *conf);

extern
int __pivot_index(__armas_dense_t *A, armas_conf_t *conf);

// internal householder
extern
void __compute_householder(__armas_dense_t *a11, __armas_dense_t *x,
                           __armas_dense_t *tau, armas_conf_t *conf);
extern
void __compute_householder_vec(__armas_dense_t *x, __armas_dense_t *tau, armas_conf_t *conf);

extern
void __compute_householder_rev(__armas_dense_t *x, __armas_dense_t *tau, armas_conf_t *conf);

extern
int __apply_householder2x1(__armas_dense_t *tau, __armas_dense_t *v,
                           __armas_dense_t *a1,  __armas_dense_t *A2,
                           __armas_dense_t *w1,  int flags, armas_conf_t *conf);

extern
int __apply_householder1x1(__armas_dense_t *tau, __armas_dense_t *v,
                           __armas_dense_t *a1,  __armas_dense_t *A2,
                           __armas_dense_t *w1,  int flags, armas_conf_t *conf);


// internal LDL functions
extern int
__unblk_bkfactor_lower(__armas_dense_t *A, __armas_dense_t *W, armas_pivot_t *P, armas_conf_t *conf);
extern int
__unblk_bkfactor_upper(__armas_dense_t *A, __armas_dense_t *W, armas_pivot_t *P, armas_conf_t *conf);
extern int
__unblk_bksolve_lower(__armas_dense_t *B, __armas_dense_t *A, armas_pivot_t *P, int phase, armas_conf_t *conf);
extern int
__unblk_bksolve_upper(__armas_dense_t *B, __armas_dense_t *A, armas_pivot_t *P, int phase, armas_conf_t *conf);
extern int
__blk_bkfactor_lower(__armas_dense_t *A, __armas_dense_t *W, armas_pivot_t *P, int lb, armas_conf_t *conf);
extern int
__blk_bkfactor_upper(__armas_dense_t *A, __armas_dense_t *W, armas_pivot_t *P, int lb, armas_conf_t *conf);

// internal LQ functions
extern int
__update_lq_right(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		  __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_lq_left(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		 __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_lq_reflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
		     armas_conf_t *conf);

// internal QL functions
extern int
__update_ql_right(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		  __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_ql_left(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		 __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_ql_reflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
		     armas_conf_t *conf);

// internal QR functions
extern int
__update_qr_right(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		  __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_qr_left(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		 __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_qr_reflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
		     armas_conf_t *conf);
// internal RQ functions
extern int
__update_rq_right(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		  __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_rq_left(__armas_dense_t *C1, __armas_dense_t *C2, __armas_dense_t *Y1,
		 __armas_dense_t *Y2, __armas_dense_t *T, __armas_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_rq_reflector(__armas_dense_t *T, __armas_dense_t *A, __armas_dense_t *tau,
		     armas_conf_t *conf);




#endif /* __ARMAS_INTERNAL_LAPACK_H */
