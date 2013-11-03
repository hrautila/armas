#!/bin/bash

SIZES="200 400 600 800 1000 1200 1400 1600"
NPROC=$(echo $(grep processor /proc/cpuinfo | wc -l)/2 | bc)

while [ true ]
do
	case $1 in
		-P) NPROC=$2; shift;;
		*) break;;
	esac
	shift
done

prog=$1

MODEL=$(grep 'model name' /proc/cpuinfo | tail -1 | sed 's/ *model name.*: *//')
echo "CPU : " $MODEL
echo "Test: " $prog $NPROC CPUs
for N in $SIZES
do
	./$prog -P $NPROC $N
done


