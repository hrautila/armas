
AUTOMAKE_OPTIONS = subdir-objects
INCLUDE = -I$(top_srcdir)/src/include -I$(top_srcdir)/tests
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall $(INCLUDE)

noinst_HEADERS = ../helper.h

if ENABLE_FLOAT32
LIBARMAS = $(top_builddir)/src/libarmass.la
AM_CPPFLAGS = $(INCLUDE) -DFLOAT32
endif

if ENABLE_FLOAT64
LIBARMAS = $(top_builddir)/src/libarmasd.la
AM_CPPFLAGS = $(INCLUDE) -DFLOAT64
endif

if ENABLE_COMPLEX64
LIBARMAS = $(top_builddir)/src/libarmasc.la
AM_CPPFLAGS = $(INCLUDE) -DCOMPLEX64
endif

if ENABLE_COMPLEX128
LIBARMAS = $(top_builddir)/src/libarmasz.la
AM_CPPFLAGS = $(INCLUDE) -DCOMPLEX128
endif

UNIT_TESTS = \
	sort chol givens house housem ldlbk ldlt lu qrbld qrfact qrmult \
	qrsol qlfact qlbld qlmult lqfact lqbld lqmult lqsol \
	rqfact rqbld rqmult hessred trdred bidiag bdsvd trdevd \
	evd gesvd rbt secular invtrm inv invspd invldlt

check_PROGRAMS = $(UNIT_TESTS)

bdsvd_SOURCES  = bdsvd.c ../helper.c
bdsvd_LDADD    = $(LIBARMAS)
bidiag_SOURCES = bidiag.c ../helper.c
bidiag_LDADD   = $(LIBARMAS)
evd_SOURCES    = evd.c ../helper.c
evd_LDADD      = $(LIBARMAS)
chol_SOURCES   = chol.c ../helper.c
chol_LDADD     = $(LIBARMAS)
gesvd_SOURCES  = gesvd.c ../helper.c
gesvd_LDADD    = $(LIBARMAS)
givens_SOURCES = givens.c ../helper.c
givens_LDADD   = $(LIBARMAS)
house_SOURCES  = house.c ../helper.c
house_LDADD    = $(LIBARMAS)
housem_SOURCES = housem.c ../helper.c
housem_LDADD   = $(LIBARMAS)
inv_SOURCES    = inv.c ../helper.c
inv_LDADD      = $(LIBARMAS)
invspd_SOURCES = invspd.c ../helper.c
invspd_LDADD   = $(LIBARMAS)
invldlt_SOURCES = invldlt.c ../helper.c
invldlt_LDADD  = $(LIBARMAS)
invtrm_SOURCES = invtrm.c ../helper.c
invtrm_LDADD   = $(LIBARMAS)
ldlbk_SOURCES  = ldlbk.c ../helper.c
ldlbk_LDADD    = $(LIBARMAS)
ldlt_SOURCES   = ldlt.c ../helper.c
ldlt_LDADD     = $(LIBARMAS)
lu_SOURCES     = lu.c ../helper.c
lu_LDADD       = $(LIBARMAS)
lqbld_SOURCES  = lqbld.c ../helper.c
lqbld_LDADD    = $(LIBARMAS)
lqfact_SOURCES = lqfact.c ../helper.c
lqfact_LDADD   = $(LIBARMAS)
lqmult_SOURCES = lqmult.c ../helper.c
lqmult_LDADD   = $(LIBARMAS)
lqsol_SOURCES  = lqsol.c ../helper.c
lqsol_LDADD    = $(LIBARMAS)
qrbld_SOURCES  = qrbld.c ../helper.c
qrbld_LDADD    = $(LIBARMAS)
qrfact_SOURCES = qrfact.c ../helper.c
qrfact_LDADD   = $(LIBARMAS)
qrmult_SOURCES = qrmult.c ../helper.c
qrmult_LDADD   = $(LIBARMAS)
qrsol_SOURCES  = qrsol.c ../helper.c
qrsol_LDADD    = $(LIBARMAS)
qlbld_SOURCES  = qlbld.c ../helper.c
qlbld_LDADD    = $(LIBARMAS)
qlfact_SOURCES = qlfact.c ../helper.c
qlfact_LDADD   = $(LIBARMAS)
qlmult_SOURCES = qlmult.c ../helper.c
qlmult_LDADD   = $(LIBARMAS)
rbt_SOURCES    = rbt.c ../helper.c
rbt_LDADD      = $(LIBARMAS)
rqbld_SOURCES  = rqbld.c ../helper.c
rqbld_LDADD    = $(LIBARMAS)
rqfact_SOURCES = rqfact.c ../helper.c
rqfact_LDADD   = $(LIBARMAS)
rqmult_SOURCES = rqmult.c ../helper.c
rqmult_LDADD   = $(LIBARMAS)
hessred_SOURCES = hessred.c ../helper.c
hessred_LDADD  = $(LIBARMAS)
trdevd_SOURCES = trdevd.c ../helper.c
trdevd_LDADD   = $(LIBARMAS)
trdred_SOURCES = trdred.c ../helper.c
trdred_LDADD   = $(LIBARMAS)
secular_SOURCES = secular.c ../helper.c
secular_LDADD  = $(LIBARMAS)
sort_SOURCES = sort.c ../helper.c
sort_LDADD  = $(LIBARMAS)

TESTS = $(check_PROGRAMS)

