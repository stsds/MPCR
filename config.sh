#!/bin/bash

##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MMPR is an R package provided by the STSDS group at KAUST
##########################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_SOURCE_DIR=$(dirname "$0")

if [[ "$OSTYPE" == "darwin"* ]]; then
  ABSOLUE_PATH=$([[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}")
else
  ABSOLUE_PATH=$(dirname $(realpath "$0"))
fi

while getopts ":f:c:tevhi:" opt; do
  case $opt in
  f) ##### Define test file path  #####
    echo -e "${BLUE}Test File path set to $OPTARG${NC}"
    TEST_PATH=$OPTARG
    ;;
  c) ##### Define test file path  #####
      echo -e "${BLUE}Config File path set to $OPTARG${NC}"
      CONFIG_PATH=$OPTARG
      ;;
  t) ##### Building tests enabled #####
    echo -e "${GREEN}Building tests enabled${NC}"
    BUILDING_TESTS="ON"
    ;;
  e) ##### Building examples enabled #####
    echo -e "${GREEN}Building examples enabled${NC}"
    BUILDING_EXAMPLES="ON"
    ;;
  i) ##### Define installation path  #####
    echo -e "${BLUE}Installation path set to $OPTARG${NC}"
    INSTALL_PATH=$OPTARG
    ;;
  v) ##### printing full output of make #####
    echo -e "${YELLOW}printing make with details${NC}"
    VERBOSE=ON
    ;;
  \?) ##### using default settings #####
    echo -e "${RED}Building tests disabled${NC}"
    echo -e "${RED}Building examples disabled${NC}"
    echo -e "${BLUE}Installation path set to /usr/local${NC}"
    echo -e "${BLUE}Test File path set to tests/test-files${NC}"
    BUILDING_EXAMPLES="OFF"
    BUILDING_TESTS="OFF"
    INSTALL_PATH="/usr/local"
    VERBOSE=OFF
    TEST_PATH="${ABSOLUE_PATH}/tests/test-files"
    CONFIG_PATH="${ABSOLUE_PATH}/config"
    ;;
  :) ##### Error in an option #####
    echo "Option $OPTARG requires parameter(s)"
    exit 0
    ;;
  h) ##### Prints the help #####
    echo "Usage of $(basename "$0"):"
    echo ""
    printf "%20s %s\n" "-t :" "to enable building tests"
    echo ""
    printf "%20s %s\n" "-e :" "to enable building examples"
    echo ""
    printf "%20s %s\n" "-i [path] :" "specify installation path"
    printf "%20s %s\n" "" "default = /usr/local"
    printf "%20s %s\n" "-f [path] :" "specify test files path"
    printf "%20s %s\n" "" "default = tests/test-files"
    echo ""
    exit 1
    ;;
  esac
done

if [ -z "$BUILDING_TESTS" ]; then
  BUILDING_TESTS="OFF"
  echo -e "${RED}Building tests disabled${NC}"
fi

if [ -z "$BUILDING_EXAMPLES" ]; then
  BUILDING_EXAMPLES="OFF"
  echo -e "${RED}Building examples disabled${NC}"
fi

if [ -z "$INSTALL_PATH" ]; then
  INSTALL_PATH="/usr/local"
  echo -e "${BLUE}Installation path set to $INSTALL_PATH${NC}"
fi

if [ -z "$TEST_PATH" ]; then
  TEST_PATH="${ABSOLUE_PATH}/tests/test-files"
  echo -e "${BLUE}Test Files path set to $TEST_PATH${NC}"
fi
if [ -z "$CONFIG_PATH" ]; then
  CONFIG_PATH="${ABSOLUE_PATH}/config"
  echo -e "${BLUE}Configuration Files path set to $CONFIG_PATH${NC}"
fi

rm -rf bin/
mkdir bin/
cmake -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMPR_BUILD_TESTS=$BUILDING_TESTS \
  -DMPR_BUILD_EXAMPLES=$BUILDING_EXAMPLES \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
  -DCMAKE_TEST_PREFIX="$TEST_PATH" \
  -DCMAKE_CONFIG_PREFIX="$CONFIG_PATH" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=$VERBOSE \
  -H"${PROJECT_SOURCE_DIR}" \
  -B"${PROJECT_SOURCE_DIR}/bin" \
  -DRUNNING_CPP=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DUSE_CUDA=OFF \
  -DCMAKE_CXX_FLAGS_DEBUG="-fPIC" \
  -DCMAKE_CXX_FLAGS_RELEASE="-fPIC"
