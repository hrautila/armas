
##DEBUG = -D__DEBUG__
DEBUG =
AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -g -O3 -march=native -ffast-math -fomit-frame-pointer -pthread -Iinc $(DEBUG)

noinst_HEADERS = \
	inc/dtype.h \
	inc/dlpack.h \
	inc/internal.h \
	inc/internal_lapack.h \
	inc/scheduler.h \
	inc/sync.h \
	inc/chan.h \
	inc/compat.h \
	inc/kernel.h \
	inc/matrix.h \
	inc/mult.h \
	inc/mult_nosimd.h \
	inc/mult_sse_f64.h \
	inc/mult_avx_f64.h \
	inc/mult_fma_f64.h \
	inc/mvec_nosimd.h \
	inc/pivot.h \
	inc/partition.h \
	inc/mult_avx_c128.h \
	inc/mult_avx_c64.h \
	inc/mult_avx_f32.h \
	inc/mult_fma_c128.h \
	inc/mult_fma_c64.h \
	inc/mult_fma_f32.h \
	inc/mult_sse_c128.h \
	inc/mult_sse_c64.h \
	inc/mult_sse_f32.h 


COMMON = \
	core/support.c \
	core/sched.c \
	core/pivot_ext.c

CORE = \
	core/kernel.c \
	core/matrix.c \
	core/matrix_ext.c \
	core/mnorm.c \
	core/mscale.c \
	core/plus.c \
	core/partition.c \
	core/pivot.c 



BLAS1 = \
	blas1/asum.c \
	blas1/axpy.c \
	blas1/copy.c \
	blas1/dot.c \
	blas1/iamax.c \
	blas1/nrm2.c \
	blas1/scal.c \
	blas1/swap.c

BLAS2 = \
	blas2/gemv.c \
	blas2/symv.c \
	blas2/ger.c \
	blas2/syr2.c \
	blas2/syr.c \
	blas2/trmv.c \
	blas2/trsv.c \
	blas2/updtrmv.c 

BLAS3 = \
	blas3/gemm.c \
	blas3/symm.c \
	blas3/syr2k.c \
	blas3/syrk.c \
	blas3/trmm_blk.c \
	blas3/trmm.c \
	blas3/trmm_rec.c \
	blas3/trmm_unb.c \
	blas3/trsm_blk.c \
	blas3/trsm.c \
	blas3/trsm_rec.c \
	blas3/trsm_unb.c \
	blas3/updtrmm.c

LAPACK = \
	lapack/evd.c \
	lapack/svd.c \
	lapack/trdsec.c \
	lapack/trdevd.c \
	lapack/sort.c \
	lapack/bdsvd.c \
	lapack/bdsvd_gr.c \
	lapack/bdsvd_dk.c \
	lapack/aux.c \
	lapack/givens.c \
	lapack/diag.c \
	lapack/qdroot.c \
	lapack/bired.c \
	lapack/bibld.c \
	lapack/chol.c \
	lapack/hess.c \
	lapack/house.c \
	lapack/ldlbk.c \
	lapack/ldlbkl.c \
	lapack/ldlbku.c \
	lapack/lq.c \
	lapack/lqbld.c \
	lapack/lqmult.c \
	lapack/lqsolve.c \
	lapack/lu.c \
	lapack/qlbld.c \
	lapack/ql.c \
	lapack/qlmult.c \
	lapack/qrbld.c \
	lapack/qr.c \
	lapack/qrmult.c \
	lapack/qrsolve.c \
	lapack/rqbld.c \
	lapack/rq.c \
	lapack/rqmult.c \
	lapack/support.c \
	lapack/trired.c \
	lapack/tribld.c


COMPAT = \
	compat/axpy.c \
	compat/copy.c \
	compat/dot.c \
	compat/nrm2.c \
	compat/swap.c \
	compat/gemv.c \
	compat/symv.c \
	compat/ger.c \
	compat/syr.c \
	compat/syr2.c \
	compat/gemm.c \
	compat/symm.c \
	compat/potrf.c 

noinst_LTLIBRARIES = libd.la libz.la libx.la libdcompat.la

# common to all 
libx_la_SOURCES = $(COMMON)

# double precision
libd_la_SOURCES = $(CORE) $(BLAS1) $(BLAS2) $(BLAS3) $(LAPACK)
libd_la_CPPFLAGS = -DFLOAT64 -DWITH_AVX

# double precision compability
libdcompat_la_SOURCES = $(COMPAT)
libdcompat_la_CPPFLAGS = -DFLOAT64 -DCOMPAT

# complex support
libz_la_SOURCES = $(CORE) $(BLAS1) $(BLAS2)
libz_la_CPPFLAGS = -DCOMPLEX128

lib_LTLIBRARIES = libarmasd.la libarmasz.la
libarmasd_la_SOURCES = 
libarmasd_la_LIBADD = libx.la libd.la libdcompat.la

libarmasz_la_SOURCES = 
libarmasz_la_LIBADD = libx.la libz.la


BUILT_SOURCES = armas/dmatrix.h armas/zmatrix.h armas/linalgd.h
CLEANFILES = armas/dmatrix.h armas/zmatrix.h armas/linalgd.h

armas/dmatrix.h: inc/matrix.h
	sed 's/__armas_/armas_d_/g;s/_ARMAS_MAT/_ARMAS_DMAT/;s/DTYPE/double/g;s/ABSTYPE/double/g;s/__ZERO/0.0/g;s:/\* COMPLEX_H \*/::' $< > $@

armas/linalgd.h: inc/linalg.h
	sed 's/__armas_/armas_d_/g;s/_ARMAS_MAT/_ARMAS_DMAT/;s/DTYPE/double/g;s/ABSTYPE/double/g;s:"matrix.h":<armas/dmatrix.h>:;s/_LINALG_/_LINALGD_/;' $< > $@

# comples types

armas/zmatrix.h: inc/matrix.h
	sed 's/__armas_/armas_z_/g;s/_ARMAS_MAT/_ARMAS_ZMAT/;s/DTYPE/double complex/g;s/ABSTYPE/double/g;s/__ZERO/0.0+0.0i/g;s:/\* COMPLEX_H \*/:#include <complex.h>:' $< > $@


