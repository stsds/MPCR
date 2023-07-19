#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "Requires 1 parameters : output file path"
  exit 1
fi

ABSOLUE_PATH=$(dirname $(realpath "$0"))

#row=(100 1000 10000 100000)
#col=(100 1000 10000 100000)
row=(50000)
col=(50000)

function run_MMPR_benchmark() {
#  Rscript ${ABSOLUE_PATH}/chol.R $1 $3 $4 >>$5
#  Rscript ${ABSOLUE_PATH}/crossprod.R $1 $2 $3 $4 >>$5
  #Rscript ${ABSOLUE_PATH}/eigen.R $1 $3 $4 >>$5
#  Rscript ${ABSOLUE_PATH}/norm.R $1 $3 $4 >>$5
#  Rscript ${ABSOLUE_PATH}/qr.R $1 $3 $4 >>$5
#  Rscript ${ABSOLUE_PATH}/rcond.R $1 $2 $3 $4 >>$5
#  Rscript ${ABSOLUE_PATH}/solve.R $1 $3 $4 >>$5
  Rscript ${ABSOLUE_PATH}/triangularsolve.R $1 $3 $4 >>$5
#  Rscript ${ABSOLUE_PATH}/svd.R $1 $3 $4 >>$5
}

echo "Running MMPR benchmark" >>$1
for i in {0..1}; do
  echo "----------------------------------------------------------------------------------------------------" >>$1
  echo "----------------------------------------------------------------------------------------------------" >>$1
  run_MMPR_benchmark ${row[i]} ${col[i]} 3 1 $1
done
