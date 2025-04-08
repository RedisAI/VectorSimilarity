#!/bin/bash
version=2024.1
processor=$(uname -m)
OS_TYPE=$(uname -s)
MODE=$1 # whether to install using sudo or not

if [[ $OS_TYPE = 'Linux' ]]
then
    if [[ $processor = 'x86_64' ]]
    then
        wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | $MODE tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
        echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | $MODE tee /etc/apt/sources.list.d/intel-oneapi.list
        $MODE apt update -qq || true
        $MODE apt install -yqq intel-oneapi-mkl-devel-${version}
    fi
fi
