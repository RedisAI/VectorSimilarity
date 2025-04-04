#!/bin/bash
MODE=$1 # whether to install using sudo or not
set -e
export DEBIAN_FRONTEND=noninteractive
$MODE dnf update -y
$MODE dnf install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ make wget git valgrind

cp /opt/rh/gcc-toolset-13/enable /etc/profile.d/gcc-toolset-13.sh

$MODE dnf install -y intel-oneapi-mkl-devel

source install_cmake.sh $MODE
