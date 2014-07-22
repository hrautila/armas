
AUTOMAKE_OPTIONS = subdir-objects

AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -I../ -I../../src

LDADD = ../../src/libarmasd.la

noinst_HEADERS = ../helper.h

noinst_PROGRAMS = dgemm dsymm dtrmm dtrsm dsymv dgemv dtrmv dger dsyr2

dgemm_SOURCES = perfgemm.c ../helper.c

dsymm_SOURCES = perfsymm.c ../helper.c

dtrmm_SOURCES = perftrmm.c ../helper.c

dtrsm_SOURCES = perftrsm.c ../helper.c

dsymv_SOURCES = perfsymv.c ../helper.c

dgemv_SOURCES = perfgemv.c ../helper.c

dger_SOURCES  = perfger.c ../helper.c

dtrmv_SOURCES = perftrmv.c ../helper.c

dsyr2_SOURCES = perfsyr2.c ../helper.c

