
AUTOMAKE_OPTIONS = subdir-objects
INCLUDE = -I$(top_srcdir)/src/include -I$(top_srcdir)/tests
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall $(INCLUDE)


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

