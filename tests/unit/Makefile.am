
AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -I../ -I../../src -I../../src/inc

LDADD = ../../src/libarmasd.la

noinst_HEADERS = ../helper.h

DOUBLE = dgemm dsymm dtrmm dtrsm dsyrk dtrmrk dgemv dger dtrmv dtrsv diag

LAPACK =  \
	evd gesvd secular \
	trdevd bdsvd givens trdred \
	bidiag chol hess lu ldl \
	qrfact qrbld qrmult qrsol \
	lqfact lqbld lqmult lqsol \
	qlfact qlbld qlmult \
	rqfact rqbld rqmult

if HAVE_GMP_MPFR
EXT = ext_gemm ext_trmm ext_trsm ext_trsv

# EXT PRECISION
ext_gemm_SOURCES = ext_gemm.c ../helper.c ext_prec.c
ext_gemm_LDFLAGS = -lmpfr -lgmp
ext_trmm_SOURCES = ext_trmm.c ../helper.c ext_prec.c
ext_trmm_LDFLAGS = -lmpfr -lgmp
ext_trsm_SOURCES = ext_trsm.c ../helper.c ext_prec.c
ext_trsm_LDFLAGS = -lmpfr -lgmp
ext_trsv_SOURCES = ext_trsv.c ../helper.c ext_prec.c
ext_trsv_LDFLAGS = -lmpfr -lgmp
endif


check_PROGRAMS = $(DOUBLE) $(LAPACK) $(EXT)

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
diag_SOURCES   = diag.c ../helper.c


## LAPACK
evd_SOURCES    = evd.c ../helper.c
gesvd_SOURCES  = gesvd.c ../helper.c
secular_SOURCES = secular.c ../helper.c
trdevd_SOURCES = trdevd.c ../helper.c
bdsvd_SOURCES  = bdsvd.c ../helper.c
givens_SOURCES = givens.c ../helper.c
bidiag_SOURCES = bidiag.c ../helper.c
chol_SOURCES   = chol.c ../helper.c
lu_SOURCES     = lu.c ../helper.c
ldl_SOURCES    = ldl.c ../helper.c
hess_SOURCES   = hessred.c ../helper.c
trdred_SOURCES = trdred.c ../helper.c

qrfact_SOURCES = qrfact.c ../helper.c
qrbld_SOURCES  = qrbld.c ../helper.c
qrmult_SOURCES = qrmult.c ../helper.c
qrsol_SOURCES  = qrsol.c ../helper.c

qlfact_SOURCES = qlfact.c ../helper.c
qlbld_SOURCES  = qlbld.c ../helper.c
qlmult_SOURCES = qlmult.c ../helper.c

lqfact_SOURCES = lqfact.c ../helper.c
lqbld_SOURCES  = lqbld.c ../helper.c
lqmult_SOURCES = lqmult.c ../helper.c
lqsol_SOURCES  = lqsol.c ../helper.c

rqfact_SOURCES = rqfact.c ../helper.c
rqbld_SOURCES  = rqbld.c ../helper.c
rqmult_SOURCES = rqmult.c ../helper.c

TESTS = $(check_PROGRAMS)




