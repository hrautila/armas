
AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -I../ -I../../src

LDADD = ../../src/libarmasd.la

noinst_HEADERS = ../helper.h

DOUBLE = dgemm dsymm dtrmm dtrsm dsyrk dtrmrk dgemv dger dtrmv dtrsv

LAPACK =  bidiag chol hess lu ldl qrfact qrbld qrmult lqfact lqbld lqmult qlfact qlbld qlmult

check_PROGRAMS = $(DOUBLE) $(LAPACK)

# BLAS
dgemm_SOURCES  = gemm.c ../helper.c
dsymm_SOURCES  = symm.c ../helper.c
dtrmm_SOURCES  = trmm.c ../helper.c
dtrsm_SOURCES  = trsm.c ../helper.c
dsyrk_SOURCES  = syrk.c ../helper.c
dtrmrk_SOURCES = trmrk.c ../helper.c
dgemv_SOURCES  = gemv.c ../helper.c
dger_SOURCES   = ger.c ../helper.c
dtrmv_SOURCES  = trmv.c ../helper.c
dtrsv_SOURCES  = trsv.c ../helper.c

## LAPACK
bidiag_SOURCES = bidiag.c ../helper.c
chol_SOURCES   = chol.c ../helper.c
lu_SOURCES     = lu.c ../helper.c
ldl_SOURCES    = ldl.c ../helper.c
hess_SOURCES   = hessred.c ../helper.c

qrfact_SOURCES = qrfact.c ../helper.c
qrbld_SOURCES  = qrbld.c ../helper.c
qrmult_SOURCES = qrmult.c ../helper.c

qlfact_SOURCES = qlfact.c ../helper.c
qlbld_SOURCES  = qlbld.c ../helper.c
qlmult_SOURCES = qlmult.c ../helper.c

lqfact_SOURCES = lqfact.c ../helper.c
lqbld_SOURCES  = lqbld.c ../helper.c
lqmult_SOURCES = lqmult.c ../helper.c


TESTS = $(check_PROGRAMS)




