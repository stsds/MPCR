#!/bin/bash

ABSOLUE_PATH=$(dirname $(realpath "$0"))

if
  R_Lib=$(Rscript $ABSOLUE_PATH/cmake/FindRLibraryPath.R)
then
  echo "R Library Path : " $R_Lib
  else
    echo "Error Getting R Library Path, Make sure .libPaths() in R works fine or set ENV R_LIB_PATH manually. "
fi

export R_LIB_PATH=$R_Lib:$R_LIB_PATH
source "$ABSOLUE_PATH/env.sh"

if
  R CMD check .
then
  R CMD build .
  Rscript $ABSOLUE_PATH/tests/R-tests/InstallPackage.R
fi
