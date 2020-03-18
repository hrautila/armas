
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
	efttest ext_gemm

check_PROGRAMS = $(UNIT_TESTS)

efttest_SOURCES   = efttest.c
efttest_LDADD     = $(LIBARMAS)
ext_gemm_SOURCES  = ext_gemm.c ../helper.c ../helper_ext.c
ext_gemm_LDADD    = $(LIBARMAS)

TESTS = $(check_PROGRAMS)




