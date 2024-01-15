#!/bin/bash
MODE=$1 # whether to install using sudo or not
set -e
export DEBIAN_FRONTEND=noninteractive

$MODE tdnf install -q -y build-essential git wget

source install_cmake.sh $MODE
