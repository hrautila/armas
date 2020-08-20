#!/bin/sh
CMDDIR=$(dirname $0)
if [ -d $CMDDIR/.git ]; then
    VERSION=$(git describe --always)
    DIRTY=$(git diff-index --name-only HEAD)
    test "${DIRTY}" != "" && VERSION="${VERSION}-dirty"
elif [ -f $CMDDIR/.version ]; then
    VERSION=$(cat .version)
else
    VERSION="unknown"
fi

printf '%s' "$VERSION"
