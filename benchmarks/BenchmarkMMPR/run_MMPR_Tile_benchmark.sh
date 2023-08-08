#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "Requires 1 parameters : output file path"
  exit 1
fi

ABSOLUE_PATH=$(dirname $(realpath "$0"))

row=(50000)
col=(50000)
tile_row=(500)
tile_col=(500)
num_threads=(1 4 8 16 32)

function run_MMPR_benchmark() {
  Rscript ${ABSOLUE_PATH}/tile_gemm.R $1 $2 $3 $4 $5 $6 $7 >>$8
  Rscript ${ABSOLUE_PATH}/tile_chol.R $1 $3 $5 $6 $7 >>$8
  Rscript ${ABSOLUE_PATH}/tile_trsm.R $1 $3 $5 $6 >>$8
}

echo "Running MMPR benchmark" >>$1
for i in {0..5}; do
  echo "----------------------------------------------------------------------------------------------------" >>$1
  echo "----------------------------------------------------------------------------------------------------" >>$1
  run_MMPR_benchmark ${row[i]} ${col[i]} ${tile_row[i]} ${tile_col[i]} 3 1 ${num_threads[i]} $1
done
