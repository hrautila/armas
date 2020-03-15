
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
	base/system.c \
	base/matrix.c \
	base/matrix_ext.c \
	base/partition.c \
	base/pivot.c \
	base/pivot_ext.c \
	io/iomm.c \
	io/json_base.c \
	io/json_file.c \
	io/json_reader.c \
	io/json_writer.c

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
	blas/swap.c \
	blas/symm.c \
	blas/symv.c \
	blas/syr.c \
	blas/syr2.c \
	blas/syr2k.c \
	blas/syrk.c \
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
	blas/updtrmm.c \
	blas/updtrmv.c

EBLAS =
if CONFIG_EXT_PRECISION
EBLAS += \
	eblas/asum_ext.c \
	eblas/axpy_ext.c \
	eblas/dot_ext.c \
	eblas/gemm_ext.c \
	eblas/gemv_ext.c \
	eblas/ger_ext.c \
	eblas/kernel_ext.c \
	eblas/scale.c \
	eblas/symv_ext.c \
	eblas/syr2_ext.c \
	eblas/syr_ext.c \
	eblas/trmm_ext.c \
	eblas/trmv_ext.c \
	eblas/trsm_ext.c \
	eblas/trsv_ext.c
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

LAPACK = \
	lapack/aux.c \
	lapack/bdsweep.c \
	lapack/bdsvd.c \
	lapack/bdsvd_dk.c \
	lapack/bdsvd_gr.c \
	lapack/bibld.c \
	lapack/bired.c \
	lapack/biredm.c \
	lapack/chol.c \
	lapack/cholpv.c \
	lapack/cholupd.c \
	lapack/diag.c \
	lapack/dqds.c \
	lapack/evd.c \
	lapack/givens.c \
	lapack/hess.c \
	lapack/hessm.c \
	lapack/house.c \
	lapack/housem.c \
	lapack/hhouse.c \
	lapack/inv.c \
	lapack/invldl.c \
	lapack/invspd.c \
	lapack/invtrm.c \
	lapack/ldlbk.c \
	lapack/ldlbkl.c \
	lapack/ldlbku.c \
	lapack/ldl.c \
	lapack/ldlpv.c \
	lapack/lqbld.c \
	lapack/lq.c \
	lapack/lqmult.c \
	lapack/lqsolve.c \
	lapack/lu.c \
	lapack/qdroot.c \
	lapack/qlbld.c \
	lapack/ql.c \
	lapack/qlmult.c \
	lapack/qrbld.c \
	lapack/qr.c \
	lapack/qrmult.c \
	lapack/qrsolve.c \
	lapack/qrt.c \
	lapack/rbtgm.c \
	lapack/rbtgu.c \
	lapack/rotate.c \
	lapack/rqbld.c \
	lapack/rq.c \
	lapack/rqmult.c \
	lapack/sort.c \
	lapack/trdbis.c \
	lapack/trdevd.c \
	lapack/trdsec.c \
	lapack/trdsweep.c \
	lapack/tribld.c \
	lapack/trired.c \
	lapack/triredm.c \
	lapack/svd.c

if OLD
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
libd_la_SOURCES = $(COMMON) $(BLAS) $(EBLAS) $(ACCEL) $(COMPAT) $(LAPACK)
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
