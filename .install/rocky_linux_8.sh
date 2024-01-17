#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not

$MODE yum install -y gcc-toolset-10-gcc gcc-toolset-10-gcc-c++ make valgrind wget git
$MODE source /opt/rh/gcc-toolset-10/enable
$MODE update-alternatives --install /usr/bin/gcc gcc /opt/rh/gcc-toolset-10/root/usr/bin/gcc 60 \
                          --slave   /usr/bin/g++ g++ /opt/rh/gcc-toolset-10/root/usr/bin/g++
source install_cmake.sh $MODE
