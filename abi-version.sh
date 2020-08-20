#!/bin/sh

CMDDIR=$(dirname $0)

ARMAS_H="${CMDDIR}/src/include/armas.h"
if [ ! -f "$ARMAS_H" ]; then
    echo "abi-version.sh: header armas.h not available"
    exit 1
fi

CURRENT=$(awk '/^#define +ARMAS_ABI_CURRENT/ {print $3}' $ARMAS_H)
REVISION=$(awk '/^#define +ARMAS_ABI_REVISION/ {print $3}' $ARMAS_H)
AGE=$(awk '/^#define +ARMAS_ABI_AGE/ {print $3}' $ARMAS_H)

if [ -z "$CURRENT" -o -z "$REVISION" -o -z "$AGE" ]; then
    echo "abi-version.sh: abi version macros not found."
    exit 1
fi

case $1 in
    -libtool)
        printf '%s' "$CURRENT:$REVISION:$AGE"
        ;;
    *)
        printf '%s' "$CURRENT.$REVISION.$AGE"
        ;;
esac

