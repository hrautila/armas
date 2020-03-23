
AUTOMAKE_OPTIONS = subdir-objects

INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/tests
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall $(INCLUDES)


noinst_HEADERS = ../helper.h

if ENABLE_FLOAT32
LIBARMAS = $(top_builddir)/src/libarmass.la
endif

if ENABLE_COMPLEX64
LIBARMAS = $(top_builddir)/src/libarmasc.la
endif

if ENABLE_COMPLEX128
LIBARMAS = $(top_builddir)/src/libarmasz.la
endif

if ENABLE_FLOAT64
LIBARMAS = $(top_builddir)/src/libarmasd.la
endif


UNIT_TESTS = \
	efttest ext_dot ext_gemm ext_trmm ext_trsm ext_gemv ext_trmv ext_trsv

check_PROGRAMS = $(UNIT_TESTS)

efttest_SOURCES   = efttest.c
efttest_LDADD     = $(LIBARMAS)
ext_dot_SOURCES   = ext_dot.c ../helper.c ../helper_ext.c
ext_dot_LDADD     = $(LIBARMAS)
ext_gemm_SOURCES  = ext_gemm.c ../helper.c ../helper_ext.c
ext_gemm_LDADD    = $(LIBARMAS)
ext_gemv_SOURCES  = ext_gemv.c ../helper.c ../helper_ext.c
ext_gemv_LDADD    = $(LIBARMAS)
ext_trmm_SOURCES  = ext_trmm.c ../helper.c ../helper_ext.c
ext_trmm_LDADD    = $(LIBARMAS)
ext_trmv_SOURCES  = ext_trmv.c ../helper.c ../helper_ext.c
ext_trmv_LDADD    = $(LIBARMAS)
ext_trsm_SOURCES  = ext_trsm.c ../helper.c ../helper_ext.c
ext_trsm_LDADD    = $(LIBARMAS)
ext_trsv_SOURCES  = ext_trsv.c ../helper.c ../helper_ext.c
ext_trsv_LDADD    = $(LIBARMAS)

TESTS = $(check_PROGRAMS)




