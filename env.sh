#!/bin/bash

#Run source env.sh from Project root dir

export LD_LIBRARY_PATH=/usr/local/lib64/R/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib64/R/:$LD_LIBRARY_PATH
export RDIR=/usr/local/lib64/R/

export MPR_DIR=$(pwd)