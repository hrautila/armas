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

# Checks for programs.
AC_PROG_CC
##AC_PROG_RANLIB


# Checks for libraries.
AC_CHECK_LIB([m], [fabs])

# Checks for header files.
AC_CHECK_HEADERS([float.h stdint.h stdlib.h string.h sys/time.h unistd.h pthread.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_INT64_T
AC_C_RESTRICT
AC_CHECK_HEADER_STDBOOL
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([gettimeofday sqrt])

AC_CONFIG_FILES([Makefile
	src/Makefile
        tests/Makefile
	tests/unit/Makefile
	tests/perf/Makefile])
AC_OUTPUT
