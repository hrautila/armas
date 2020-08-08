
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
	base/pivot.c \
	base/pivot_ext.c \
	io/iomm.c \
	io/json_base.c \
	io/json_file.c \
	io/json_reader.c \
	io/json_writer.c

BLAS = \
	base/matrix.c \
	base/matrix_ext.c \
	base/partition.c \
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
	eblas/symm_ext.c \
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
COMMON += \
	base/accel.c

ACCEL += \
	threaded/entry.c \
	threaded/gemm.c \
	threaded/symm.c \
	threaded/trmm.c \
	threaded/trsm.c \
	workers/entry.c \
	workers/workers.c \
	workers/gemm.c
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

SPARSE =
if CONFIG_SPARSE
SPARSE += \
	sparse/accum.c \
	sparse/add.c \
	sparse/cgne.c \
	sparse/cgnr.c \
	sparse/cgrad.c \
	sparse/convert.c \
	sparse/copy.c \
	sparse/dense.c \
	sparse/gemv.c \
	sparse/gmres.c \
	sparse/icholz.c \
	sparse/iluz.c \
	sparse/io.c \
	sparse/mult.c \
	sparse/pcgrad.c \
	sparse/pgmres.c \
	sparse/sort.c \
	sparse/sparse.c \
	sparse/symv.c \
	sparse/trmv.c \
	sparse/trsv.c \
	sparse/util.c
endif

COMPAT =
if CONFIG_COMPAT
COMPAT += \
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
	compat/syr2k.c
endif

noinst_LTLIBRARIES =

# common to all
lib_LTLIBRARIES =

##libx_la_SOURCES = $(COMMON)
BUILT_SOURCES = armas/armas.h

armas/armas.h: include/armas.h
	$(top_srcdir)/install-sh $< $@

## ----------------- double precission -------------------------
if ENABLE_FLOAT64

lib_LTLIBRARIES += libarmasd.la
noinst_LTLIBRARIES += libx.la libdb.la libda.la libds.la libdc.la
libarmasd_la_SOURCES =
libarmasd_la_LIBADD = libdb.la libx.la libda.la libds.la libdc.la
libarmasd_la_LDFLAGS = -version-info @SOVERSION@

BUILT_SOURCES += armas/ddense.h armas/dsparse.h

libx_la_SOURCES = $(COMMON)
libdb_la_SOURCES = $(BLAS) $(EBLAS) $(LAPACK)
libda_la_SOURCES = $(ACCEL)
libds_la_SOURCES = $(SPARSE)
libdc_la_SOURCES = $(COMPAT)

libdb_la_CPPFLAGS = -DFLOAT64
libda_la_CPPFLAGS = -DFLOAT64
libdc_la_CPPFLAGS = -DFLOAT64
libds_la_CPPFLAGS = -DFLOAT64

armas/ddense.h: include/matrix.h
	$(top_srcdir)/src/make-header.sh $< $@ double

armas/dsparse.h: include/sparse.h
	$(top_srcdir)/src/make-header.sh $< $@ double

endif

## ----------------- single precission -------------------------
if ENABLE_FLOAT32

endif

## ----------------- complex double precission -------------------------
if ENABLE_COMPLEX128

endif

nobase_include_HEADERS = $(BUILT_SOURCES)

CLEANFILES = $(BUILT_SOURCES)

noinst_HEADERS = \
	include/partition.h								\
	include/arm/mult_armneon_f32.h					\
	include/arm/eftops.h							\
	include/arm/simd_ops.h							\
	include/arm/mult_armneon_fma_f32.h				\
	include/matcpy.h								\
	include/eft.h									\
	include/complex32/names_blas.h					\
	include/complex32/names_base.h					\
	include/complex32/names_lapack.h				\
	include/complex64/names_blas.h					\
	include/complex64/names_base.h					\
	include/complex64/names_lapack.h				\
	include/kernel_ext.h							\
	include/internal_lapack.h						\
	include/x86_64/mult_sse_c64.h					\
	include/x86_64/mult_fma_c64.h					\
	include/x86_64/mult_avx_f32.h					\
	include/x86_64/mult_fma_f64.h					\
	include/x86_64/mult_ext_avx_f32.h				\
	include/x86_64/mult_fma_f32.h					\
	include/x86_64/mult_fma_c128.h					\
	include/x86_64/mult_avx_c64.h					\
	include/x86_64/mult_sse_f64.h					\
	include/x86_64/eftops.h							\
	include/x86_64/mult_avx_c128.h					\
	include/x86_64/mult_sse_f32.h					\
	include/x86_64/simd_ops.h						\
	include/x86_64/mult_ext_avx_f64.h				\
	include/x86_64/mult_avx_f64.h					\
	include/x86_64/mult_sse_c128.h					\
	include/simd.h									\
	include/internal.h								\
	include/pivot.h									\
	include/cache.h									\
	include/float64/names_compat.h					\
	include/float64/names_blas.h					\
	include/float64/names_base.h					\
	include/float64/names_sparse.h					\
	include/float64/names_lapack.h					\
	include/compat.h								\
	include/dlpack.h								\
	include/dtype.h									\
	include/nosimd/mult.h							\
	include/nosimd/mult_ext.h						\
	include/nosimd/mvec.h							\
	include/spdefs.h								\
	include/float32/names_blas.h					\
	include/float32/names_base.h					\
	include/float32/names_lapack.h					\
	include/kernel.h								\
	include/accel.h


