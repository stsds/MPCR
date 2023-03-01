#!/bin/bash

#Run source env.sh from Project root dir

export MPR_DIR=$(pwd)
export MPR_PACKAGE_DIR=$(pwd)/MPR_1.0.tar.gz

export LD_LIBRARY_PATH=$(pwd)/bin/_deps/lapackpp-build/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/bin/_deps/blaspp-build/:$LD_LIBRARY_PATH
