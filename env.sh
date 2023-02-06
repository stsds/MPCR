#!/bin/bash

#Run source env.sh from Project root dir

export LD_LIBRARY_PATH=/usr/local/lib64/R/lib/:$LD_LIBRARY_PATH
export R_HOME=/usr/local/lib64/R/
export R_LIB_PATH=/usr/local/lib64/R/library/

export MPR_DIR=$(pwd)
export MPR_PACKAGE_DIR=$(pwd)/MPR_1.0.tar.gz