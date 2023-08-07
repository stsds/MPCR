#!/bin/bash

# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.

verbose=
num_proc="-j $(nproc)"

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

cd bin/ || exit
make clean
make all $num_proc $verbose
