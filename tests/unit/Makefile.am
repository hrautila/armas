
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -I../ -I../../src

LDADD = ../../src/libarmasd.la

noinst_HEADERS = ../helper.h

DOUBLE = dgemm dsymm dtrmm dtrsm dsyrk dtrmrk dgemv dger dtrmv dtrsv

check_PROGRAMS = $(DOUBLE)

dgemm_SOURCES = gemm.c ../helper.c
dsymm_SOURCES = symm.c ../helper.c
dtrmm_SOURCES = trmm.c ../helper.c
dtrsm_SOURCES = trsm.c ../helper.c
dsyrk_SOURCES = syrk.c ../helper.c
dtrmrk_SOURCES = trmrk.c ../helper.c
dgemv_SOURCES = gemv.c ../helper.c
dger_SOURCES  = ger.c ../helper.c
dtrmv_SOURCES = trmv.c ../helper.c
dtrsv_SOURCES = trsv.c ../helper.c

TESTS = $(check_PROGRAMS)




