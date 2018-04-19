#!/bin/bash

SRC=$1
DST=$2
TYPE=$3

DTYPE=$TYPE
ABSTYPE=$TYPE

case "$TYPE" in
    double)
	ZERO=0.0
	ABSZERO=0.0
	ONE=1.0
	COMPLEX_H=
	PREFIX=armas_d_
	PREFIX_SP=armassp_d_
	MATRIX_DEF=__ARMAS_DMATRIX_H
	LINALG_DEF=__ARMAS_LINALGD_H
	SPARSE_DEF=__ARMAS_DSPARSE_H
	;;
    float)
	ZERO=0.0
	ABSZERO=0.0
	ONE=1.0
	COMPLEX_H=
	PREFIX=armas_s_
	PREFIX_SP=armassp_s_
	MATRIX_DEF=__ARMAS_SMATRIX_H
	LINALG_DEF=__ARMAS_LINALGS_H
	SPARSE_DEF=__ARMAS_SSPARSE_H
	;;
    complex)
	ZERO=0.0+0.0i
	ABSZERO=0.0
	ONE=1.0+0.0i
	COMPLEX_H="#include <complex.h>"
	DTYPE="complex"
	ABSTYPE=float
	PREFIX=armas_c_
	PREFIX_SP=armassp_c_
	MATRIX_DEF=__ARMAS_CMATRIX_H
	LINALG_DEF=__ARMAS_LINALGC_H
	SPARSE_DEF=__ARMAS_CSPARSE_H
	;;
    zcomplex)
	ZERO=0.0+0.0i
	ABSZERO=0.0
	ONE=1.0+0.0i
	COMPLEX_H="#include <complex.h>"
	DTYPE="double complex"
	ASBTYPE=double
	PREFIX=armas_z_
	PREFIX_SP=armassp_z_
	MATRIX_DEF=__ARMAS_ZMATRIX_H
	LINALG_DEF=__ARMAS_LINALGZ_H
	SPARSE_DEF=__ARMAS_ZSPARSE_H
	;;
    *)
	;;   
esac

SUBST="\
s/armas_x_/$PREFIX/g;\
s/armassp_x_/$PREFIX_SP/g;\
s/__ARMAS_MATRIX_H/$MATRIX_DEF/;\
s/__ARMAS_LINALG_H/$LINALG_DEF/;\
s/__ARMAS_SPARSE_H/$SPARSE_DEF/;\
s/DTYPE/$DTYPE/g;\
s/ABSTYPE/$ABSTYPE/g;\
s/__ZERO/$ZERO/;\
s/__ABSZERO/$ABSZERO/;\
s/__ONE/$ONE/;\
s:\"matrix.h\":<$DST>:;\
s:/\* COMPLEX_H \*/:$COMPLEX_H:;"

echo sed-command "$SUBST"

sed "$SUBST" <$SRC >$DST

	
