#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([Another Rewrite of Matrix Algebra Subroutines], [0.5a], [],
	[armas], [https://github.com/hrautila/armas])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/Makefile.am])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
LT_INIT

## option --enable-threads
AC_ARG_ENABLE(threads,
   AC_HELP_STRING([--disable-threads],[make a threading (default is yes)]),
      [enable_threads=$enableval], [enable_threads=yes])

AC_ARG_ENABLE(ext-precision,
   AC_HELP_STRING([--disable-ext-precision], [enable extended precision (default is yes)]),
      [enable_ext_precision=$enableval], [enable_ext_precision=yes])

## option --enable-float32
AC_ARG_ENABLE(float32,
   AC_HELP_STRING([--enable-float32],[enable single precission float (default is no)]),
      [enable_float32=$enableval], [enable_float32=no])
## option --enable-float64
AC_ARG_ENABLE(float64,
   AC_HELP_STRING([--disable-float64],[enable double precission float (default is yes)]),
      [enable_float64=$enableval], [enable_float64=yes])
## option --enable-complex64
AC_ARG_ENABLE(complex64,
   AC_HELP_STRING([--enable-complex64],[enable single precission complex (default is no)]),
      [enable_complex64=$enableval], [enable_complex64=no])
## option --enable-complex128
AC_ARG_ENABLE(complex128,
   AC_HELP_STRING([--enable-complex128],[enable double precission complex (default is no)]),
      [enable_complex128=$enableval], [enable_complex128=no])

## option --enable-compat
AC_ARG_ENABLE(compat,
   AC_HELP_STRING([--enable-compat],[enable Fortan callable compability library (default is no)]),
      [enable_compat=$enableval], [enable_compat=no]) 

# Checks for programs.
AC_PROG_CC
##AC_PROG_RANLIB

# Checks for libraries.
AC_CHECK_LIB([m], [fabs])

# Checks for header files.
AC_CHECK_HEADERS([float.h math.h stdint.h stdlib.h string.h sys/time.h unistd.h pthread.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_INT64_T
AC_C_RESTRICT
AC_CHECK_HEADER_STDBOOL
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([gettimeofday sqrt])

## Threads require linux/futex.h for mutual exclusion primitives
AS_IF([test "$enable_threads" != "no"],
  [AC_CHECK_HEADER(linux/futex.h,
      [enable_threads=yes
       AC_DEFINE([ENABLE_THREADS], [1], [Enable treaded version])],
      [enable_threads=no
       AC_MSG_NOTICE(Creating non-threaded version)])],
  [:])

AS_IF([test "$enable_ext_precision" = "yes"],
  AC_DEFINE([EXT_PRECISION], [1], [Enable extended precision]), [])

## extended precision tests need libgmp and libmpfr
AC_CHECK_HEADERS([gmp.h mpfr.h],
  [AC_CHECK_LIB([mpfr],[mpfr_init2],
     [AC_CHECK_LIB([gmp], [__gmp_printf],
        [have_gmp_and_mpfr=yes], [have_gmp_and_mpfr=no])],
     [have_gmp_and_mpfr=no])],
  [have_gmp_and_mpfr=no])

## enable threading in Makefile
AM_CONDITIONAL([ENABLE_THREADS], [test "$enable_threads" = "yes"])

## enable floating point types
AM_CONDITIONAL([ENABLE_FLOAT64],    [test "$enable_float64" = "yes"])
AM_CONDITIONAL([ENABLE_FLOAT32],    [test "$enable_float32" = "yes"])
AM_CONDITIONAL([ENABLE_COMPLEX64],  [test "$enable_complex64" = "yes"])
AM_CONDITIONAL([ENABLE_COMPLEX128], [test "$enable_complex128" = "yes"])
AM_CONDITIONAL([ENABLE_EXT_PRECISION], [test "$enable_ext_precision" = "yes"])
AM_CONDITIONAL([ENABLE_COMPAT],     [test "$enable_compat" = "yes"])

AM_CONDITIONAL([HAVE_GMP_MPFR], [test "$have_gmp_and_mpfr" = "yes"])

## print out configuration options

AC_MSG_NOTICE([ENABLE_FLOAT32 = $enable_float32])
AC_MSG_NOTICE([ENABLE_FLOAT64 = $enable_float64])
AC_MSG_NOTICE([ENABLE_COMPLEX64 = $enable_complex64])
AC_MSG_NOTICE([ENABLE_COMPLEX128 = $enable_complex128])
AC_MSG_NOTICE([ENABLE_THREADS = $enable_threads])
AC_MSG_NOTICE([ENABLE_EXT_PRECISION = $enable_ext_precision])
AC_MSG_NOTICE([ENABLE_COMPAT = $enable_compat])
AC_MSG_NOTICE([HAVE_GMP_MPFR = $have_gmp_and_mpfr])

AC_CONFIG_FILES([Makefile
	src/Makefile
        tests/Makefile
	tests/unit/Makefile
	tests/perf/Makefile])
AC_OUTPUT
