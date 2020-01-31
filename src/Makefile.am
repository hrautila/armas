
##DEBUG = -D__DEBUG__
DEBUG =
AUTOMAKE_OPTIONS = subdir-objects

if CONFIG_ACCELERATORS
USE_PTHREAD=-pthread
else
USE_PTHREAD=
endif

AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall $(USE_PTHREAD) -I$(top_srcdir)/src/include $(DEBUG)

COMMON = \
	base/armas.c \
	base/cache.c \
	base/env.c \
	base/system.c
	base/matrix.c \
	base/matrix_ext.c

BLAS = \
	blas/asum.c \
	blas/axpy.c \
	blas/dot.c \
	blas/gemm.c \
	blas/gemv.c \
	blas/ger.c \
	blas/iamax.c \
	blas/kernel.c \
	blas/madd.c \
	blas/mcopy.c \
	blas/mnorm.c \
	blas/mplus.c \
	blas/mscale.c \
	blas/mset.c \
	blas/nrm2.c \
	blas/symm.c \
	blas/syr.c \
	blas/syr2.c \
	blas/trmm.c \
	blas/trmm_blk.c \
	blas/trmm_rec.c \
	blas/trmm_unb.c \
	blas/trmv.c \
	blas/trsm.c \
	blas/trsm_blk.c \
	blas/trsm_rec.c \
	blas/trsm_unb.c \
	blas/trsv.c \
	blas/updtrmv.c \
	blas/updtrmm.c \
	blas/syrk.c \
	blas/syr2k.c

EBLAS =
if CONFIG_EXT_PRECISION
EBLAS += \
	eblas/kernel_ext.c \
	eblas/gemm_ext.c \
	eblas/scale.c \
	eblas/dot_ext.c \
	eblas/asum_ext.c \
	eblas/axpy_ext.c \
	eblas/gemv_ext.c \
	eblas/trmm_ext.c \
	eblas/trsm_ext.c \
	eblas/trmv_ext.c \
	eblas/trsv_ext.c \
	eblas/symv_ext.c \
	eblas/syr2_ext.c \
	eblas/syr_ext.c \
	eblas/ger_ext.c
endif

ACCEL =
if CONFIG_ACCELERATORS
ACCEL += \
	threaded/entry.c \
	threaded/gemm.c \
	threaded/symm.c \
	threaded/trmm.c \
	threaded/trsm.c
endif

COMPAT =
if CONFIG_COMPAT
endif

if OLD
noinst_HEADERS = \
	inc/cond.h \
	inc/dtype.h \
	inc/dlpack.h \
	inc/internal.h \
	inc/internal_lapack.h \
	inc/scheduler.h \
	inc/eft.h \
	inc/eft_arm.h \
	inc/eft_x86_64.h \
	inc/sync.h \
	inc/chan.h \
	inc/compat.h \
	inc/kernel.h \
	inc/kernel_ext.h \
	inc/matrix.h \
	inc/mult.h \
	inc/mult_armneon_f32.h \
	inc/mult_armneon_fma_f32.h \
	inc/mult_nosimd.h \
	inc/mult_sse_f64.h \
	inc/mult_avx_f32.h \
	inc/mult_avx_f64.h \
	inc/mult_fma_f32.h \
	inc/mult_fma_f64.h \
	inc/mvec_nosimd.h \
	inc/pivot.h \
	inc/partition.h \
	inc/simd.h \
	inc/mult_avx_c128.h \
	inc/mult_avx_c64.h \
	inc/mult_avx_f32.h \
	inc/mult_fma_c128.h \
	inc/mult_fma_c64.h \
	inc/mult_fma_f32.h \
	inc/mult_sse_c128.h \
	inc/mult_sse_c64.h \
	inc/mult_sse_f32.h \
	inc/mult_ext_avx_f32.h \
	inc/mult_ext_avx_f64.h \
	inc/mult_ext_nosimd.h


OLD_COMMON = \
	core/armas.c \
	core/support.c \
	core/sched.c \
	core/cache.c \
	core/pivot_ext.c \
	core/json_reader.c \
	core/json_writer.c

OLD_CORE = \
	core/matrix.c \
	core/matrix_ext.c \
	core/mnorm.c \
	core/mscale.c \
	core/plus.c \
	core/partition.c \
	core/pivot.c \
	core/madd.c \
	core/msub.c \
	core/mmul.c \
	core/mdiv.c \
	core/mapply.c \
	core/iomm.c \
	core/json_base.c \
	core/json_file.c

OLD_BLAS1 = \
	blas1/asum.c \
	blas1/axpy.c \
	blas1/copy.c \
	blas1/dot.c \
	blas1/iamax.c \
	blas1/nrm2.c \
	blas1/scal.c \
	blas1/swap.c

OLD_XBLAS1 = \
	blas1/asum_ext.c \
	blas1/dot_ext.c \
	blas1/axpy_ext.c

OLD_BLAS2 = \
	blas2/gemv.c \
	blas2/symv.c \
	blas2/ger.c \
	blas2/syr2.c \
	blas2/syr.c \
	blas2/trmv.c \
	blas2/trsv.c \
	blas2/updtrmv.c 

OLD_XBLAS2 = \
	blas2/gemv_ext.c \
	blas2/symv_ext.c \
	blas2/trsv_ext.c \
	blas2/trmv_ext.c \
	blas2/update_ext.c

OLD_BLAS3 = \
	blas3/kernel.c \
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

OLD_XBLAS3 = \
	blas3/kernel_ext.c \
	blas3/symm_ext.c \
	blas3/trmm_ext.c \
	blas3/trsm_ext.c \
	blas3/updtrm_ext.c \
	blas3/syrk_ext.c

LAPACK = \
	lapack/housem.c \
	lapack/hhouse.c \
	lapack/trdbis.c \
	lapack/cholupd.c \
	lapack/cholpv.c \
	lapack/invldl.c \
	lapack/ldl.c \
	lapack/ldlpv.c \
	lapack/inv.c \
	lapack/invtrm.c \
	lapack/invspd.c \
	lapack/rbtgm.c \
	lapack/rbtgu.c \
	lapack/house.c \
	lapack/support.c \
	lapack/qr.c \
	lapack/qrmult.c \
	lapack/qrbld.c  \
	lapack/qrsolve.c  \
	lapack/ql.c \
	lapack/qlmult.c \
	lapack/qlbld.c \
	lapack/lq.c \
	lapack/lqmult.c \
	lapack/lqbld.c \
	lapack/lqsolve.c \
	lapack/rq.c \
	lapack/rqmult.c \
	lapack/rqbld.c \
	lapack/ldlbk.c \
	lapack/ldlbkl.c \
	lapack/ldlbku.c \
	lapack/lu.c \
	lapack/chol.c \
	lapack/givens.c \
	lapack/qdroot.c \
	lapack/diag.c \
	lapack/sort.c \
	lapack/bired.c \
	lapack/bibld.c \
	lapack/trired.c \
	lapack/tribld.c \
	lapack/hess.c \
	lapack/bdsvd.c \
	lapack/bdsvd_gr.c \
	lapack/bdsvd_dk.c \
	lapack/aux.c \
	lapack/trdsec.c \
	lapack/trdevd.c \
	lapack/evd.c \
	lapack/svd.c \
	lapack/qrt.c \
	lapack/dqds.c

SPARSE = \
	sparse/iluz.c \
	sparse/icholz.c \
	sparse/dense.c \
	sparse/mult.c \
	sparse/sparse.c \
	sparse/accum.c \
	sparse/add.c \
	sparse/cgrad.c \
	sparse/pcgrad.c \
	sparse/cgnr.c \
	sparse/cgne.c \
	sparse/convert.c \
	sparse/copy.c \
	sparse/gemv.c \
	sparse/gmres.c \
	sparse/pgmres.c \
	sparse/io.c \
	sparse/sort.c \
	sparse/symv.c \
	sparse/trmv.c \
	sparse/trsv.c \
	sparse/util.c

OLD_COMPAT = \
	compat/asum.c \
	compat/axpy.c \
	compat/copy.c \
	compat/dot.c \
	compat/iamax.c \
	compat/nrm2.c \
	compat/swap.c \
	compat/rot.c \
	compat/scal.c \
	compat/gemv.c \
	compat/symv.c \
	compat/trmv.c \
	compat/trsv.c \
	compat/ger.c \
	compat/syr.c \
	compat/syr2.c \
	compat/gemm.c \
	compat/symm.c \
	compat/trmm.c \
	compat/trsm.c \
	compat/syrk.c \
	compat/syr2k.c \
	compat/potrf.c
endif

##noinst_LTLIBRARIES = libd.la libz.la libx.la libdext.la libdcompat.la
noinst_LTLIBRARIES =

# common to all
lib_LTLIBRARIES =

libx_la_SOURCES = $(COMMON)
BUILT_SOURCES =

## ----------------- double precission -------------------------
if ENABLE_FLOAT64

lib_LTLIBRARIES += libarmasd.la
noinst_LTLIBRARIES += libd.la
libarmasd_la_SOURCES =
libarmasd_la_LIBADD = libd.la


# double precision
libd_la_SOURCES = $(COMMON) $(BLAS) $(EBLAS) $(ACCEL) $(COMPAT)
libd_la_CPPFLAGS = -DFLOAT64 

# # sparse double precision
# libspd_la_SOURCES = $(SPARSE)
# libspd_la_CPPFLAGS = -DFLOAT64

# if CONFIG_EXT_PRECISION
# # extended precision
# libdext_la_SOURCES = $(EBLAS)
# libdext_la_CPPFLAGS = -DFLOAT64 
# noinst_LTLIBRARIES += libdext.la
# libarmasd_la_LIBADD += libdext.la
# endif

# if CONFIG_COMPAT
# # double precision compability
# libdcompat_la_SOURCES = $(COMPAT)
# libdcompat_la_CPPFLAGS = -DFLOAT64 -DCOMPAT
# noinst_LTLIBRARIES += libdcompat.la
# libarmasd_la_LIBADD += libdcompat.la
# endif

#BUILT_SOURCES += armas/dmatrix.h armas/linalgd.h armas/dsparse.h

endif

## ----------------- single precission -------------------------
if ENABLE_FLOAT32
lib_LTLIBRARIES += libarmass.la
noinst_LTLIBRARIES += libx.la libs.la 
libarmass_la_SOURCES =
libarmass_la_LIBADD = libx.la libs.la

# single precision
libs_la_SOURCES = $(CORE) $(BLAS1) $(BLAS2) $(BLAS3) $(LAPACK)
libs_la_CPPFLAGS = -DFLOAT32 

if CONFIG_EXT_PRECISION
# extended precision
libsext_la_SOURCES = $(XBLAS1) $(XBLAS2) $(XBLAS3)
libsext_la_CPPFLAGS = -DFLOAT32
noinst_LTLIBRARIES += libsext.la
libarmass_la_LIBADD += libsext.la
endif

if CONFIG_COMPAT
# single precision compability
libscompat_la_SOURCES = $(COMPAT)
libscompat_la_CPPFLAGS = -DFLOAT32 -DCOMPAT
noinst_LTLIBRARIES += libscompat.la
libarmass_la_LIBADD += libscompat.la
endif

#BUILT_SOURCES += armas/smatrix.h armas/linalgs.h

endif


## ----------------- complex double precission -------------------------
if ENABLE_COMPLEX128

noinst_LTLIBRARIES += libx.la libz.la
lib_LTLIBRARIES += libarmasz.la

# complex support
libz_la_SOURCES = $(COMMON) $(BLAS) $(EBLAS)
libz_la_CPPFLAGS = -DCOMPLEX128

libarmasz_la_SOURCES = 
libarmasz_la_LIBADD = libx.la libz.la


#BUILT_SOURCES += armas/zmatrix.h armas/libalgz.h

endif

nobase_include_HEADERS = armas/armas.h $(BUILT_SOURCES)

CLEANFILES = $(BUILT_SOURCES)

# armas/smatrix.h: inc/matrix.h
# 	./make-header.sh $< $@ float

# armas/linalgs.h: inc/linalg.h
# 	./make-header.sh $< $@ float

# armas/dmatrix.h: inc/matrix.h
# 	./make-header.sh $< $@ double

# armas/linalgd.h: inc/linalg.h
# 	./make-header.sh $< $@ double

# armas/dsparse.h: sparse/sparse.h
# 	./make-header.sh $< $@ double

# # comples types

# armas/zmatrix.h: inc/matrix.h
# 	./make-header.sh $< $@ zcomplex


###	sed 's/armas_x_/armas_d_/g;s/_ARMAS_MAT/_ARMAS_DMAT/;s/DTYPE/double/g;s/ABSTYPE/double/g;s/__ZERO/0.0/g;s:/\* COMPLEX_H \*/::' $< > $@
