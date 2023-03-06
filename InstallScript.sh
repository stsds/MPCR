#!/bin/bash

ABSOLUE_PATH=$(dirname $(realpath "$0"))

if
  R_Lib=$(Rscript $ABSOLUE_PATH/cmake/FindRLibraryPath.R)
then
  echo "Rcpp Library Path : " $R_Lib
else
  echo "Error Getting Rcpp installation Path , Make sure Rcpp is installed. "
fi

export R_LIB_PATH=$R_Lib:$R_LIB_PATH
source "$ABSOLUE_PATH/env.sh"

if
  R CMD check .
then
  R CMD build .
  if
    Rscript $ABSOLUE_PATH/tests/R-tests/InstallPackage.R
  then
    echo "Package installed successfully !"
  else
    echo "error while installing package"
  fi
fi
