#!/bin/bash


BLAS_DIR=${MKLROOT}/lib/intel64
OPENMP_DIR=${CMPLR_ROOT}/linux/compiler/lib/intel64_lin


while getopts "blas:openmp:" opt; do
  case $opt in
    blas)
      BLAS_DIR="$OPTARG"
      ;;
    openmp)
      OPENMP_DIR="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
          echo "Option -$OPTARG requires an argument." >&2
          exit 1
          ;;
  esac
done








export LD_PRELOAD=${BLAS_DIR}/libmkl_rt.so
export LD_PRELOAD=${BLAS_DIR}/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=${BLAS_DIR}/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=${OPENMP_DIR}/libiomp5.so:$LD_PRELOAD



echo $LD_PRELOAD