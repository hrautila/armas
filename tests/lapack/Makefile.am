
AUTOMAKE_OPTIONS = subdir-objects
INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/tests
AM_CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -Wall $(INCLUDES)


noinst_HEADERS = ../helper.h

if ENABLE_FLOAT32
LIBARMAS = $(top_builddir)/src/libarmass.la
endif

if ENABLE_COMPLEX64
LIBARMAS = $(top_builddir)/src/libarmasc.la
endif

if ENABLE_COMPLEX128
LIBARMAS = $(top_builddir)/src/libarmasz.la
endif

if ENABLE_FLOAT64
LIBARMAS = $(top_builddir)/src/libarmasd.la
endif

UNIT_TESTS = \
	chol givens house housem ldlbk ldlt

check_PROGRAMS = $(UNIT_TESTS)

chol_SOURCES   = chol.c ../helper.c
chol_LDADD     = $(LIBARMAS)
givens_SOURCES = givens.c ../helper.c
givens_LDADD   = $(LIBARMAS)
house_SOURCES  = house.c ../helper.c
house_LDADD    = $(LIBARMAS)
housem_SOURCES = housem.c ../helper.c
housem_LDADD   = $(LIBARMAS)
ldlbk_SOURCES  = ldlbk.c ../helper.c
ldlbk_LDADD    = $(LIBARMAS)
ldlt_SOURCES   = ldlt.c ../helper.c
ldlt_LDADD     = $(LIBARMAS)

TESTS = $(check_PROGRAMS)

