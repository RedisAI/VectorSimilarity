#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not

$MODE apt-get update -qq || true
$MODE apt-get install -yqq gcc-12 g++-12 git wget build-essential valgrind lcov
$MODE update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12
# align gcov version with gcc version
$MODE update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-12 60

# Instruction to install Intel MKL (required for SVS) from https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | $MODE tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | $MODE tee /etc/apt/sources.list.d/oneAPI.list
$MODE apt update
$MODE apt install -y intel-oneapi-mkl

source install_cmake.sh $MODE
