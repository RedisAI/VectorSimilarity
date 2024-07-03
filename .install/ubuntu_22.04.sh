#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not

$MODE apt-get update -qq || true
$MODE apt-get install -yqq gcc-12 g++-12 git wget build-essential valgrind lcov
$MODE update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12
source install_cmake.sh $MODE
