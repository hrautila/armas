
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -I../ -I../../src

LDADD = ../../src/libarmasd.la

noinst_HEADERS = ../helper.h

noinst_PROGRAMS = dgemm dsymm dtrmm dtrsm

dgemm_SOURCES = perfgemm.c ../helper.c

dsymm_SOURCES = perfsymm.c ../helper.c

dtrmm_SOURCES = perftrmm.c ../helper.c

dtrsm_SOURCES = perftrsm.c ../helper.c

