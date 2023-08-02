#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "Requires 1 parameters : output file path"
  exit 1
fi

ABSOLUE_PATH=$(dirname $(realpath "$0"))

#row=(100 1000 10000 100000)
#col=(100 1000 10000 100000)
row=(500)
col=(500)


function run_R_benchmark() {
  Rscript ${ABSOLUE_PATH}/cholRblas.R $1 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/crossprodRblas.R $1 $2 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/eigenRblas.R $1 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/normRblas.R $1 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/qrRblas.R $1 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/rcondRblas.R $1 $2 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/solveRblas.R $1 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/svdRblas.R $1 $3 $4 >>$5
  #Rscript ${ABSOLUE_PATH}/triangularsolveRblas.R $1 $3 $4 >>$5
  
}

echo "Running R benchmark" >>$1
for i in {0..1}; do
  echo "----------------------------------------------------------------------------------------------------" >>$1
  echo "----------------------------------------------------------------------------------------------------" >>$1
  run_R_benchmark ${row[i]} ${col[i]} 3 1 $1
done
