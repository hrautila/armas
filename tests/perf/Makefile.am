
AUTOMAKE_OPTIONS = subdir-objects
INCLUDE = -I$(top_srcdir)/src/include -I$(top_srcdir)/tests
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall

noinst_HEADERS = ../helper.h

if ENABLE_FLOAT32
LIBARMAS = $(top_builddir)/src/libarmass.la
AM_CPPFLAGS = $(INCLUDE) -DFLOAT32
endif

if ENABLE_FLOAT64
LIBARMAS = $(top_builddir)/src/libarmasd.la
AM_CPPFLAGS = $(INCLUDE) -DFLOAT64
endif

noinst_PROGRAMS = gemm symm trmm trsm symv gemv trmv ger syr2 perfgv perfgvr

gemm_SOURCES = perfgemm.c ../helper.c
gemm_LDADD = $(LIBARMAS)

symm_SOURCES = perfsymm.c ../helper.c
symm_LDADD = $(LIBARMAS)

trmm_SOURCES = perftrmm.c ../helper.c
trmm_LDADD = $(LIBARMAS)

trsm_SOURCES = perftrsm.c ../helper.c_
trsm_LDADD = $(LIBARMAS)

symv_SOURCES = perfsymv.c ../helper.c
symv_LDADD = $(LIBARMAS)

gemv_SOURCES = perfgemv.c ../helper.c
gemv_LDADD = $(LIBARMAS)

ger_SOURCES  = perfger.c ../helper.c
ger_LDADD = $(LIBARMAS)

trmv_SOURCES = perftrmv.c ../helper.c
trmv_LDADD = $(LIBARMAS)

syr2_SOURCES = perfsyr2.c ../helper.c
syr2_LDADD = $(LIBARMAS)

perfgv_SOURCES = perfgv.c ../helper.c
perfgv_LDADD = $(LIBARMAS)

perfgvr_SOURCES = perfgvr.c ../helper.c
perggvr_LDADD = $(LIBARMAS)


