#!/bin/bash

##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

verbose=
num_proc="-j $(nproc)"

if [[ "$OSTYPE" == "darwin"* ]]; then
  ABSOLUE_PATH=$([[ $1 == /* ]] && echo "$1" || echo "$PWD/${1#./}")
else
  ABSOLUE_PATH=$(dirname $(realpath "$0"))
fi

while getopts "vj:h" opt; do
  case $opt in
  v)
    verbose="VERBOSE=1"
    echo "Using verbose mode"
    ;;
  j)
    num_proc="-j $OPTARG"
    echo "Using $OPTARG threads to build"
    ;;
  h)
    echo "Usage of $(basename "$0"):"
    echo "	to clean the bin directory then builds the code and run it "
    echo ""
    echo "-v		     : to print the output of make in details"
    echo ""
    echo "-j <thread number> : to with a specific number of threads"
    echo ""
    exit 1
    ;;
  *)
    echo "Invalid flags entered. run using the -h flag for help"
    exit 1
    ;;
  esac
done

cd "${ABSOLUE_PATH}/bin/" || exit
make clean
make all $num_proc $verbose
