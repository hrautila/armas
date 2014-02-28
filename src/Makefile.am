
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -pthread -Iinc

noinst_HEADERS = \
	inc/dtype.h \
	inc/internal.h \
	inc/matrix.h \
	inc/mult.h \
	inc/mult_nosimd.h \
	inc/mult_sse_f64.h \
	inc/mult_avx_f64.h \
	inc/mult_fma_f64.h \
	inc/mvec_nosimd.h \
	inc/pivot.h \
	inc/partition.h

COMMON = \
	core/support.c \
	core/pivot.c

CORE = \
	core/kernel.c \
	core/matrix.c \
	core/matrix_ext.c \
	core/mnorm.c \
	core/mscale.c \
	core/plus.c \
	core/partition.c


BLAS1 = \
	blas1/asum.c \
	blas1/axpy.c \
	blas1/dot.c \
	blas1/iamax.c \
	blas1/nrm2.c \
	blas1/scal.c

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


noinst_LTLIBRARIES = libd.la libz.la libx.la

# common to all 
libx_la_SOURCES = $(COMMON)

# double precision
libd_la_SOURCES = $(CORE) $(BLAS1) $(BLAS2) $(BLAS3)
libd_la_CPPFLAGS = -DFLOAT64

# complex support
libz_la_SOURCES = $(CORE) $(BLAS1) $(BLAS2)
libz_la_CPPFLAGS = -DCOMPLEX128

lib_LTLIBRARIES = libarmasd.la libarmasz.la
libarmasd_la_SOURCES = 
libarmasd_la_LIBADD = libx.la libd.la

libarmasz_la_SOURCES = 
libarmasz_la_LIBADD = libx.la libz.la


BUILT_SOURCES = armas/dmatrix.h armas/zmatrix.h
CLEANFILES = armas/dmatrix.h armas/zmatrix.h

armas/dmatrix.h: inc/matrix.h
	sed 's/__armas_/armas_d_/g;s/_ARMAS_MAT/_ARMAS_DMAT/;s/DTYPE/double/g;s/ABSTYPE/double/g;s/__ZERO/0.0/g;s:/\* COMPLEX_H \*/::' $< > $@

armas/zmatrix.h: inc/matrix.h
	sed 's/__armas_/armas_z_/g;s/_ARMAS_MAT/_ARMAS_ZMAT/;s/DTYPE/double complex/g;s/ABSTYPE/double/g;s/__ZERO/0.0+0.0i/g;s:/\* COMPLEX_H \*/:#include <complex.h>:' $< > $@

