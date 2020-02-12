
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
	gemm
    ## symm trmm trsm syrk trmrk syr2k gemv ger trmv trsv diag

check_PROGRAMS = $(UNIT_TESTS)

# Basic operations
# base_SOURCES = mbase.c ../helper.c
# base_LDADD    = $(LIBARMAS)

gemm_SOURCES  = gemm.c ../helper.c
gemm_LDADD    = $(top_builddir)/src/libarmasd.la
# symm_SOURCES  = symm.c ../helper.c
# symm_LDADD    = $(LIBARMAS)
# trmm_SOURCES  = trmm.c ../helper.c
# trmm_LDADD    = $(LIBARMAS)
# trsm_SOURCES  = trsm.c ../helper.c
# trsm_LDADD    = $(LIBARMAS)
# syrk_SOURCES  = syrk.c ../helper.c
# syrk_LDADD    = $(LIBARMAS)
# syr2k_SOURCES = syr2k.c ../helper.c
# syr2k_LDADD   = $(LIBARMAS)
# trmrk_SOURCES = trmrk.c ../helper.c
# trmrk_LDADD    = $(LIBARMAS)
# gemv_SOURCES  = gemv.c ../helper.c
# gemv_LDADD    = $(LIBARMAS)
# ger_SOURCES   = ger.c ../helper.c
# ger_LDADD    = $(LIBARMAS)
# trmv_SOURCES  = trmv.c ../helper.c
# trmv_LDADD    = $(LIBARMAS)
# trsv_SOURCES  = trsv.c ../helper.c
# trsv_LDADD    = $(LIBARMAS)
# diag_SOURCES   = diag.c ../helper.c
# diag_LDADD    = $(LIBARMAS)

TESTS = $(check_PROGRAMS)

