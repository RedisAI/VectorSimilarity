#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not

$MODE apt update -qq
$MODE apt install -yqq git wget build-essential lcov openssl libssl-dev \
    python3 python3-venv python3-dev unzip rsync clang curl gpg-agent

# Instruction to install Intel MKL (required for SVS) from https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | $MODE tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | $MODE tee /etc/apt/sources.list.d/oneAPI.list
$MODE apt update
$MODE apt install -y intel-oneapi-mkl

source install_cmake.sh $MODE
