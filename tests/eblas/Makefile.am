
AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall -I../ -I../../src -I../../src/inc


noinst_HEADERS = ../helper.h

UNIT_TESTS = 

if ENABLE_FLOAT32
DTYPE=s
else if ENABLE_COMPLEX32
DTYPE=c
else if ENABLE_COMPLEX64
DTYPE=z
else
DTYPE=d
endif


LIBARMAS = ../../src/libarmas$(DTYPE).la

UNIT_TESTS += \
	base gemm
    ## symm trmm trsm syrk trmrk syr2k gemv ger trmv trsv diag

check_PROGRAMS = $(UNIT_TESTS)

efttest_SOURCES = efttest.c

# Basic operations
base_SOURCES = mbase.c ../helper.c
base_LDADD    = $(LIBARMAS)

# BLAS (double)
gemm_SOURCES  = gemm.c ../helper.c
gemm_LDADD    = $(LIBARMAS)
symm_SOURCES  = symm.c ../helper.c
symm_LDADD    = $(LIBARMAS)
trmm_SOURCES  = trmm.c ../helper.c
trmm_LDADD    = $(LIBARMAS)
trsm_SOURCES  = trsm.c ../helper.c
trsm_LDADD    = $(LIBARMAS)
syrk_SOURCES  = syrk.c ../helper.c
syrk_LDADD    = $(LIBARMAS)
syr2k_SOURCES = syr2k.c ../helper.c
syr2k_LDADD   = $(LIBARMAS)
trmrk_SOURCES = trmrk.c ../helper.c
trmrk_LDADD    = $(LIBARMAS)
gemv_SOURCES  = gemv.c ../helper.c
gemv_LDADD    = $(LIBARMAS)
ger_SOURCES   = ger.c ../helper.c
ger_LDADD    = $(LIBARMAS)
trmv_SOURCES  = trmv.c ../helper.c
trmv_LDADD    = $(LIBARMAS)
trsv_SOURCES  = trsv.c ../helper.c
trsv_LDADD    = $(LIBARMAS)
diag_SOURCES   = diag.c ../helper.c
diag_LDADD    = $(LIBARMAS)


TESTS = $(check_PROGRAMS)




