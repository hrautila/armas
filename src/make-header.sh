#!/bin/bash

SRC=$1
DST=$2
TYPE=$3
NOTYPENAME=${4-no}

echo "** make-header.sh: $(pwd)"

DTYPE=$TYPE
ABSTYPE=$TYPE

case "$TYPE" in
    double)
	CODE=d
	ZERO=0.0
	ABSZERO=0.0
	ONE=1.0
	COMPLEX_H=
	PREFIX=armas_d_
	PREFIX_SP=armassp_d_
	MATRIX_DEF=ARMAS_DDENSE_H
	LINALG_DEF=ARMAS_LINALGD_H
	SPARSE_DEF=ARMAS_DSPARSE_H
	;;
    float)
	CODE=s
	ZERO=0.0
	ABSZERO=0.0
	ONE=1.0
	COMPLEX_H=
	PREFIX=armas_s_
	PREFIX_SP=armassp_s_
	MATRIX_DEF=ARMAS_SDENSE_H
	LINALG_DEF=ARMAS_LINALGS_H
	SPARSE_DEF=ARMAS_SSPARSE_H
	;;
    complex)
	CODE=c
	ZERO=0.0+0.0i
	ABSZERO=0.0
	ONE=1.0+0.0i
	COMPLEX_H="#include <complex.h>"
	DTYPE="complex"
	ABSTYPE=float
	PREFIX=armas_c_
	PREFIX_SP=armassp_c_
	MATRIX_DEF=ARMAS_CDENSE_H
	LINALG_DEF=ARMAS_LINALGC_H
	SPARSE_DEF=ARMAS_CSPARSE_H
	;;
    zcomplex)
	CODE=z
	ZERO=0.0+0.0i
	ABSZERO=0.0
	ONE=1.0+0.0i
	COMPLEX_H="#include <complex.h>"
	DTYPE="double complex"
	ASBTYPE=double
	PREFIX=armas_z_
	PREFIX_SP=armassp_z_
	MATRIX_DEF=ARMAS_ZDENSE_H
	LINALG_DEF=ARMAS_LINALGZ_H
	SPARSE_DEF=ARMAS_ZSPARSE_H
	;;
    *)
	;;
esac
##set -x
MATRIXH="armas/${CODE}dense.h"

if [[ "$NOTYPENAME" = "no" ]]; then

    SUBST="\
s/armas_/${PREFIX}/g;\
s/armassp_/${PREFIX_SP}/g;
s/armas_d_conf/armas_conf/g;\
s/armas_d_iostream/armas_iostream/g;"

fi


SUBST="$SUBST \
s/ARMAS_MATRIX_H/${MATRIX_DEF}/;\
s/ARMAS_LINALG_H/${LINALG_DEF}/;\
s/ARMAS_SPARSE_H/${SPARSE_DEF}/;\
s/DTYPE/${DTYPE}/g;\
s/ABSTYPE/${ABSTYPE}/g;\
s/ZERO/${ZERO}/;\
s/ABSZERO/${ABSZERO}/;\
s/ONE/${ONE}/;\
s/^__ARMAS_INLINE/extern inline/;\
s:\"matrix.h\":<${MATRIXH}>:;\
s:/\* COMPLEX_H \*/:${COMPLEX_H}:;"

#echo sed-command "$SUBST"

sed "$SUBST" <$SRC >$DST


