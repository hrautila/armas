#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([Another Rewrite of Matrix Algebra Subroutines], [0.2], [],
	[armas], [https://github.com/hrautila/armas])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/Makefile.am])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
LT_INIT

## option --enable-threads
AC_ARG_ENABLE(threads,
   AC_HELP_STRING([--disable-threads],[enable threading (default is yes)]))

AC_ARG_ENABLE(ext-precision,
   AC_HELP_STRING([--disable-ext-precision], [enable extended precision (default is yes)]))

## option --enable-float32
AC_ARG_ENABLE(float32,
   AC_HELP_STRING([--enable-float32], [enable single precission float (default is no)]))
## option --enable-float64
AC_ARG_ENABLE(float64,
   AC_HELP_STRING([--disable-float64], [enable double precission float (default is yes)]))
## option --enable-complex64
AC_ARG_ENABLE(complex64,
   AC_HELP_STRING([--enable-complex64], [enable single precission complex (default is no)]))
## option --enable-complex128
AC_ARG_ENABLE(complex128,
   AC_HELP_STRING([--enable-complex128], [enable double precission complex (default is no)]))

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

AS_IF([test "$enable_ext_presicion" != "no"],
  AC_DEFINE([EXT_PRECISION], [1], [Enable extended precision]), [])

## enable threading in Makefile
AM_CONDITIONAL([ENABLE_THREADS], [test "$enable_threads" = "yes"])
## enable floating point types
AM_CONDITIONAL([ENABLE_FLOAT64],    [test "$enable_float64" = "yes"])
AM_CONDITIONAL([ENABLE_FLOAT32],    [test "$enable_float32" = "yes"])
AM_CONDITIONAL([ENABLE_COMPLEX64],  [test "$enable_complex64" = "yes"])
AM_CONDITIONAL([ENABLE_COMPLEX128], [test "$enable_complex128" = "yes"])

AC_CONFIG_FILES([Makefile
	src/Makefile
        tests/Makefile
	tests/unit/Makefile
	tests/perf/Makefile])
AC_OUTPUT
