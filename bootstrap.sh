#!/bin/sh

test $(which libtoolize) != "" || \
    { echo "libtoolize not found, package libtool not installed?" >&2; exit 1; }

test $(which aclocal) != "" || \
    { echo "aclocal not found, package automake not installed?" >&2; exit 1; }

test $(which automake) != "" || \
    { echo "automake not found, package automake not installed?" >&2; exit 1; }

test $(which autoconf) != "" || \
    { echo "autoconf not found, package autoconf not installed?" >&2; exit 1; }

test $(which autoheader) != "" || \
    { echo "autoheader not found, package autoconf not installed?" >&2; exit 1; }

echo "Prerequisites OK. Bootstraping ..." >&2

libtoolize
aclocal
autoconf
autoheader
automake --add-missing


