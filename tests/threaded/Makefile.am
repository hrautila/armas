
AUTOMAKE_OPTIONS = subdir-objects
INCLUDE = -I$(top_srcdir)/src/include -I$(top_srcdir)/tests
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall $(INCLUDE)


noinst_HEADERS = ../helper.h

if ENABLE_FLOAT32
LIBARMAS = $(top_builddir)/src/libarmass.la
AM_CPPFLAGS = $(INCLUDE) -DFLOAT32
endif

if ENABLE_FLOAT64
LIBARMAS = $(top_builddir)/src/libarmasd.la
AM_CPPFLAGS = $(INCLUDE) -DFLOAT64
endif

if ENABLE_COMPLEX64
LIBARMAS = $(top_builddir)/src/libarmasc.la
AM_CPPFLAGS = $(INCLUDE) -DCOMPLEX64
endif

if ENABLE_COMPLEX128
LIBARMAS = $(top_builddir)/src/libarmasz.la
AM_CPPFLAGS = $(INCLUDE) -DCOMPLEX128
endif

UNIT_TESTS = gemm symm trmm trsm

check_PROGRAMS = $(UNIT_TESTS)

gemm_SOURCES  = gemm.c ../helper.c
gemm_LDADD    = $(LIBARMAS)
symm_SOURCES  = symm.c ../helper.c
symm_LDADD    = $(LIBARMAS)
trmm_SOURCES  = trmm.c ../helper.c
trmm_LDADD    = $(LIBARMAS)
trsm_SOURCES  = trsm.c ../helper.c
trsm_LDADD    = $(LIBARMAS)

TESTS = $(check_PROGRAMS)

