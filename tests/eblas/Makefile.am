
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
	efttest gemm

check_PROGRAMS = $(UNIT_TESTS)

efttest_SOURCES = efttest.c


# BLAS (double)
gemm_SOURCES  = gemm.c ../helper.c
gemm_LDADD    = $(LIBARMAS)

TESTS = $(check_PROGRAMS)




