#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
#
AC_PREREQ([2.69])
AC_INIT([Another Rewrite of Matrix Algebra Subroutines], [m4_esyscmd(./package-version.sh)],
   [harri.rautila@gmail.com], [armas], [https://github.com/hrautila/armas])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/Makefile.am])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
LT_INIT

# shared library version number
AC_SUBST(SOVERSION, [m4_esyscmd(./abi-version.sh -libtool)])

## option --enable-plugins
AC_ARG_ENABLE(accelerators,
   AC_HELP_STRING([--disable-accelerators],[make accelerators (default is yes)]),
      [enable_accelerators=$enableval], [enable_accelerators=yes])

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

AC_ARG_ENABLE(sparse,
   AC_HELP_STRING([--enable-sparse], [enable sparse iterative methods (default is no)]),
      [enable_sparse=$enableval], [enable_sparse=no])

AC_ARG_ENABLE(notypenames,
   AC_HELP_STRING([--enable-notypenames], [enable compilation without type indicators. (default is no)]),
      [enable_notypenames=$enableval], [enable_notypenames=no])

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

AS_IF([test "$enable_ext_precision" = "yes"],
  [AC_DEFINE([CONFIG_EXT_PRECISION], [1], [Enable extended precision])], [])

AS_IF([test "$enable_accelerators" = "yes"],
  [AC_DEFINE([CONFIG_ACCELERATORS], [1], [Enable accelerators])], [])

AS_IF([test "$enable_compat" = "yes"],
  [AC_DEFINE([CONFIG_COMPAT], [1], [Enable compability])], [])

AS_IF([test "$enable_sparse" = "yes"],
  [AC_DEFINE([CONFIG_SPARSE], [1], [Enable sparse methods])] , [])

AS_IF([test "$enable_notypenames" = "yes"],
  [AC_DEFINE([CONFIG_NOTYPENAMES], [1], [Enable without type indicators in names.])] , [])

AM_CONDITIONAL([CONFIG_ACCELERATORS], [test "$enable_accelerators" = "yes"])
AM_CONDITIONAL([CONFIG_EXT_PRECISION], [test "$enable_ext_precision" = "yes"])
AM_CONDITIONAL([CONFIG_COMPAT], [test "$enable_compat" = "yes"])
AM_CONDITIONAL([CONFIG_SPARSE], [test "$enable_sparse" = "yes"])
AM_CONDITIONAL([CONFIG_NOTYPENAMES], [test "$enable_notypenames" = "yes"])

## enable floating point types
AM_CONDITIONAL([ENABLE_FLOAT64],    [test "$enable_float64" = "yes"])
AM_CONDITIONAL([ENABLE_FLOAT32],    [test "$enable_float32" = "yes"])
AM_CONDITIONAL([ENABLE_COMPLEX64],  [test "$enable_complex64" = "yes"])
AM_CONDITIONAL([ENABLE_COMPLEX128], [test "$enable_complex128" = "yes"])

## print out configuration options
AC_MSG_NOTICE([ENABLE_FLOAT32 = $enable_float32])
AC_MSG_NOTICE([ENABLE_FLOAT64 = $enable_float64])
AC_MSG_NOTICE([ENABLE_COMPLEX64 = $enable_complex64])
AC_MSG_NOTICE([ENABLE_COMPLEX128 = $enable_complex128])
AC_MSG_NOTICE([CONFIG_ACCELERATORS = $enable_accelerators])
AC_MSG_NOTICE([CONFIG_EXT_PRECISION = $enable_ext_precision])
AC_MSG_NOTICE([CONFIG_COMPAT = $enable_compat])
AC_MSG_NOTICE([CONFIG_SPARSE = $enable_sparse])
AC_MSG_NOTICE([CONFIG_NOTYPENAMES = $enable_notypenames])

AC_CONFIG_FILES([
   Makefile
   src/Makefile
   tests/Makefile
   tests/blas/Makefile
   tests/eblas/Makefile
   tests/lapack/Makefile
   tests/threaded/Makefile
   tests/workers/Makefile
   docs/Makefile
])

AC_OUTPUT
