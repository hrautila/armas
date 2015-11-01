
AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall -I../ -I../../src -I../../src/inc


noinst_HEADERS = ../helper.h

UNIT_TESTS = efttest

if ENABLE_FLOAT32
##LDADD = ../../src/libarmass.la

UNIT_TESTS += sgemm ssymm strmm strsm strmrk ssyrk ssyr2k sgemv ##strmv strsv

UNIT_TESTS += \
	sqrfact sqrmult sqrbld sqrsol sqlfact sqlmult sqlbld slqfact slqbld \
	slqmult slqsol srqfact srqbld srqmult sldl slu schol sgivens sdiag \
	sbidiag strdred shess sbdsvd ssecular strdevd sevd sgesvd

# GMP/MPFR libaries exits, can test extended precision
if HAVE_GMP_MPFR
UNIT_TESTS += ext_sgemm ext_strmm ext_strsm ext_strsv
endif

endif

if ENABLE_FLOAT64
##LDADD = ../../src/libarmasd.la

UNIT_TESTS += dgemm dsymm dtrmm dtrsm dsyrk dtrmrk dsyr2k dgemv dger dtrmv dtrsv diag

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
ext_dgemm_LDADD   = ../../src/libarmasd.la
ext_dtrmm_SOURCES = ext_trmm.c ../helper.c ext_prec.c
ext_dtrmm_LDFLAGS = -lmpfr -lgmp
ext_dtrmm_LDADD   = ../../src/libarmasd.la
ext_dtrsm_SOURCES = ext_trsm.c ../helper.c ext_prec.c
ext_dtrsm_LDFLAGS = -lmpfr -lgmp
ext_dtrsm_LDADD   = ../../src/libarmasd.la
ext_dtrsv_SOURCES = ext_trsv.c ../helper.c ext_prec.c
ext_dtrsv_LDFLAGS = -lmpfr -lgmp
ext_dtrsv_LDADD   = ../../src/libarmasd.la


check_PROGRAMS = $(UNIT_TESTS)

efttest_SOURCES = efttest.c

# BLAS (double)
dgemm_SOURCES  = gemm.c ../helper.c
dgemm_LDADD    = ../../src/libarmasd.la
dsymm_SOURCES  = symm.c ../helper.c
dsymm_LDADD    = ../../src/libarmasd.la
dtrmm_SOURCES  = trmm.c ../helper.c
dtrmm_LDADD    = ../../src/libarmasd.la
dtrsm_SOURCES  = trsm.c ../helper.c
dtrsm_LDADD    = ../../src/libarmasd.la
dsyrk_SOURCES  = syrk.c ../helper.c
dsyrk_LDADD    = ../../src/libarmasd.la
dsyr2k_SOURCES = syr2k.c ../helper.c
dsyr2k_LDADD   = ../../src/libarmasd.la
dtrmrk_SOURCES = trmrk.c ../helper.c
dtrmrk_LDADD    = ../../src/libarmasd.la
dgemv_SOURCES  = gemv.c ../helper.c
dgemv_LDADD    = ../../src/libarmasd.la
dger_SOURCES   = ger.c ../helper.c
dger_LDADD    = ../../src/libarmasd.la
dtrmv_SOURCES  = trmv.c ../helper.c
dtrmv_LDADD    = ../../src/libarmasd.la
dtrsv_SOURCES  = trsv.c ../helper.c
dtrsv_LDADD    = ../../src/libarmasd.la
diag_SOURCES   = diag.c ../helper.c
diag_LDADD    = ../../src/libarmasd.la

## LAPACK
evd_SOURCES    = evd.c ../helper.c
evd_LDADD    = ../../src/libarmasd.la
gesvd_SOURCES  = gesvd.c ../helper.c
gesvd_LDADD    = ../../src/libarmasd.la
secular_SOURCES = secular.c ../helper.c
secular_LDADD    = ../../src/libarmasd.la
trdevd_SOURCES = trdevd.c ../helper.c
trdevd_LDADD    = ../../src/libarmasd.la
bdsvd_SOURCES  = bdsvd.c ../helper.c
bdsvd_LDADD    = ../../src/libarmasd.la
givens_SOURCES = givens.c ../helper.c
givens_LDADD    = ../../src/libarmasd.la
bidiag_SOURCES = bidiag.c ../helper.c
bidiag_LDADD    = ../../src/libarmasd.la
chol_SOURCES   = chol.c ../helper.c
chol_LDADD    = ../../src/libarmasd.la
lu_SOURCES     = lu.c ../helper.c
lu_LDADD    = ../../src/libarmasd.la
ldl_SOURCES    = ldl.c ../helper.c
ldl_LDADD    = ../../src/libarmasd.la
hess_SOURCES   = hessred.c ../helper.c
hess_LDADD    = ../../src/libarmasd.la
trdred_SOURCES = trdred.c ../helper.c
trdred_LDADD    = ../../src/libarmasd.la

qrfact_SOURCES = qrfact.c ../helper.c
qrfact_LDADD    = ../../src/libarmasd.la
qrbld_SOURCES  = qrbld.c ../helper.c
qrbld_LDADD    = ../../src/libarmasd.la
qrmult_SOURCES = qrmult.c ../helper.c
qrmult_LDADD    = ../../src/libarmasd.la
qrsol_SOURCES  = qrsol.c ../helper.c
qrsol_LDADD    = ../../src/libarmasd.la

qlfact_SOURCES = qlfact.c ../helper.c
qlfact_LDADD    = ../../src/libarmasd.la
qlbld_SOURCES  = qlbld.c ../helper.c
qlbld_LDADD    = ../../src/libarmasd.la
qlmult_SOURCES = qlmult.c ../helper.c
qlmult_LDADD    = ../../src/libarmasd.la

lqfact_SOURCES = lqfact.c ../helper.c
lqfact_LDADD    = ../../src/libarmasd.la
lqbld_SOURCES  = lqbld.c ../helper.c
lqbld_LDADD    = ../../src/libarmasd.la
lqmult_SOURCES = lqmult.c ../helper.c
lqmult_LDADD    = ../../src/libarmasd.la
lqsol_SOURCES  = lqsol.c ../helper.c
lqsol_LDADD    = ../../src/libarmasd.la

rqfact_SOURCES = rqfact.c ../helper.c
rqfact_LDADD    = ../../src/libarmasd.la
rqbld_SOURCES  = rqbld.c ../helper.c
rqbld_LDADD    = ../../src/libarmasd.la
rqmult_SOURCES = rqmult.c ../helper.c
rqmult_LDADD    = ../../src/libarmasd.la

# EXT PRECISION (double precision)
ext_sgemm_SOURCES = ext_gemm.c ../helper.c ext_prec.c
ext_sgemm_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
ext_sgemm_LDFLAGS = -lmpfr -lgmp
ext_sgemm_LDADD   = ../../src/libarmass.la
ext_strmm_SOURCES = ext_trmm.c ../helper.c ext_prec.c
ext_strmm_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
ext_strmm_LDFLAGS = -lmpfr -lgmp
ext_strmm_LDADD   = ../../src/libarmass.la
ext_strsm_SOURCES = ext_trsm.c ../helper.c ext_prec.c
ext_strsm_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
ext_strsm_LDFLAGS = -lmpfr -lgmp
ext_strsm_LDADD   = ../../src/libarmass.la
ext_strsv_SOURCES = ext_trsv.c ../helper.c ext_prec.c
ext_strsv_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
ext_strsv_LDFLAGS = -lmpfr -lgmp
ext_strsv_LDADD   = ../../src/libarmass.la

# BLAS (single precision)
sgemm_SOURCES  = gemm.c ../helper.c
sgemm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
sgemm_LDADD    = ../../src/libarmass.la

strmm_SOURCES  = trmm.c ../helper.c
strmm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
strmm_LDADD    = ../../src/libarmass.la
strsm_SOURCES  = trsm.c ../helper.c
strsm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
strsm_LDADD    = ../../src/libarmass.la
ssymm_SOURCES  = symm.c ../helper.c
ssymm_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
ssymm_LDADD    = ../../src/libarmass.la
ssyrk_SOURCES  = syrk.c ../helper.c
ssyrk_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
ssyrk_LDADD    = ../../src/libarmass.la
ssyr2k_SOURCES = syr2k.c ../helper.c
ssyr2k_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
ssyr2k_LDADD   = ../../src/libarmass.la
strmrk_SOURCES = trmrk.c ../helper.c
strmrk_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
strmrk_LDADD   = ../../src/libarmass.la
sgemv_SOURCES  = gemv.c ../helper.c
sgemv_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
sgemv_LDADD    = ../../src/libarmass.la
#sger_SOURCES   = ger.c ../helper.c
#strmv_SOURCES  = trmv.c ../helper.c
#strmv_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
#strsv_SOURCES  = trsv.c ../helper.c
#strsv_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)
#sdiag_SOURCES   = diag.c ../helper.c
#sdiag_CFLAGS   = -DFLOAT32 $(AM_CFLAGS)

# LAPACK (single precision)
sqrfact_SOURCES = qrfact.c ../helper.c
sqrfact_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqrfact_LDADD   = ../../src/libarmass.la
sqrmult_SOURCES = qrmult.c ../helper.c
sqrmult_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqrmult_LDADD   = ../../src/libarmass.la
sqrbld_SOURCES = qrbld.c ../helper.c
sqrbld_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqrbld_LDADD   = ../../src/libarmass.la
sqrsol_SOURCES = qrsol.c ../helper.c
sqrsol_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqrsol_LDADD   = ../../src/libarmass.la

sqlfact_SOURCES = qlfact.c ../helper.c
sqlfact_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqlfact_LDADD   = ../../src/libarmass.la
sqlmult_SOURCES = qlmult.c ../helper.c
sqlmult_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqlmult_LDADD   = ../../src/libarmass.la
sqlbld_SOURCES = qlbld.c ../helper.c
sqlbld_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sqlbld_LDADD   = ../../src/libarmass.la

slqfact_SOURCES = lqfact.c ../helper.c
slqfact_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
slqfact_LDADD   = ../../src/libarmass.la
slqmult_SOURCES = lqmult.c ../helper.c
slqmult_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
slqmult_LDADD   = ../../src/libarmass.la
slqbld_SOURCES = lqbld.c ../helper.c
slqbld_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
slqbld_LDADD   = ../../src/libarmass.la
slqsol_SOURCES = lqsol.c ../helper.c
slqsol_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
slqsol_LDADD   = ../../src/libarmass.la

srqfact_SOURCES = rqfact.c ../helper.c
srqfact_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
srqfact_LDADD   = ../../src/libarmass.la
srqmult_SOURCES = rqmult.c ../helper.c
srqmult_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
srqmult_LDADD   = ../../src/libarmass.la
srqbld_SOURCES = rqbld.c ../helper.c
srqbld_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
srqbld_LDADD   = ../../src/libarmass.la

sldl_SOURCES = ldl.c ../helper.c
sldl_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sldl_LDADD   = ../../src/libarmass.la
slu_SOURCES = lu.c ../helper.c
slu_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
slu_LDADD   = ../../src/libarmass.la
schol_SOURCES = chol.c ../helper.c
schol_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
schol_LDADD   = ../../src/libarmass.la
sgivens_SOURCES = givens.c ../helper.c
sgivens_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sgivens_LDADD   = ../../src/libarmass.la
sdiag_SOURCES = diag.c ../helper.c
sdiag_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sdiag_LDADD   = ../../src/libarmass.la
sbidiag_SOURCES = bidiag.c ../helper.c
sbidiag_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sbidiag_LDADD   = ../../src/libarmass.la
strdred_SOURCES = trdred.c ../helper.c
strdred_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
strdred_LDADD   = ../../src/libarmass.la
shess_SOURCES = hessred.c ../helper.c
shess_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
shess_LDADD   = ../../src/libarmass.la
sbdsvd_SOURCES = bdsvd.c ../helper.c
sbdsvd_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sbdsvd_LDADD   = ../../src/libarmass.la
ssecular_SOURCES = secular.c ../helper.c
ssecular_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
ssecular_LDADD   = ../../src/libarmass.la
strdevd_SOURCES = trdevd.c ../helper.c
strdevd_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
strdevd_LDADD   = ../../src/libarmass.la
sevd_SOURCES = evd.c ../helper.c
sevd_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sevd_LDADD   = ../../src/libarmass.la
sgesvd_SOURCES = gesvd.c ../helper.c
sgesvd_CFLAGS  = -DFLOAT32 $(AM_CFLAGS)
sgesvd_LDADD   = ../../src/libarmass.la


TESTS = $(check_PROGRAMS)




