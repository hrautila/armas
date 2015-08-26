
AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall -I../ -I../../src -I../../src/inc


noinst_HEADERS = ../helper.h

UNIT_TESTS =

if ENABLE_FLOAT32
LDADD = ../../src/libarmass.la

UNIT_TESTS += sgemm ssymm strmm strsm strmrk ssyrk sgemv ##strmv strsv

endif

if ENABLE_FLOAT64
LDADD = ../../src/libarmasd.la

UNIT_TESTS += dgemm dsymm dtrmm dtrsm dsyrk dtrmrk dgemv dger dtrmv dtrsv diag

UNIT_TESTS +=  \
	evd gesvd secular \
	trdevd bdsvd givens trdred \
	bidiag chol hess lu ldl \
	qrfact qrbld qrmult qrsol \
	lqfact lqbld lqmult lqsol \
	qlfact qlbld qlmult \
	rqfact rqbld rqmult

# GMP/MPFR libaries exits, can test extended precision
if HAVE_GMP_MPFR
UNIT_TESTS += ext_dgemm ext_dtrmm ext_dtrsm ext_dtrsv
endif

endif

# EXT PRECISION (double precision)
ext_dgemm_SOURCES = ext_gemm.c ../helper.c ext_prec.c
ext_dgemm_LDFLAGS = -lmpfr -lgmp
ext_dtrmm_SOURCES = ext_trmm.c ../helper.c ext_prec.c
ext_dtrmm_LDFLAGS = -lmpfr -lgmp
ext_dtrsm_SOURCES = ext_trsm.c ../helper.c ext_prec.c
ext_dtrsm_LDFLAGS = -lmpfr -lgmp
ext_dtrsv_SOURCES = ext_trsv.c ../helper.c ext_prec.c
ext_dtrsv_LDFLAGS = -lmpfr -lgmp


check_PROGRAMS = $(UNIT_TESTS)

# BLAS (double)
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

# BLAS (single precision)
sgemm_SOURCES  = gemm.c ../helper.c
sgemm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
ssymm_SOURCES  = symm.c ../helper.c
ssymm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
strmm_SOURCES  = trmm.c ../helper.c
strmm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
strsm_SOURCES  = trsm.c ../helper.c
strsm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
ssyrk_SOURCES  = syrk.c ../helper.c
ssyrk_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
strmrk_SOURCES = trmrk.c ../helper.c
strmrk_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
sgemv_SOURCES  = gemv.c ../helper.c
sgemv_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
#sger_SOURCES   = ger.c ../helper.c
#strmv_SOURCES  = trmv.c ../helper.c
#strmv_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
#strsv_SOURCES  = trsv.c ../helper.c
#strsv_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
#sdiag_SOURCES   = diag.c ../helper.c
#sdiag_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)

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




