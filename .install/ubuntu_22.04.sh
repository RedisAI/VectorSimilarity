#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not

$MODE apt-get update -qq || true
$MODE apt-get install -yqq gcc-12 g++-12 git wget build-essential lcov
$MODE update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12
# align gcov version with gcc version
$MODE update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-12 60

# Add Clang 18 installation
# Download and run the official LLVM installation script
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
$MODE ./llvm.sh 18
$MODE apt-get install -yqq clang-18 clang++-18
$MODE update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-18

echo `clang --version`
source install_cmake.sh $MODE
