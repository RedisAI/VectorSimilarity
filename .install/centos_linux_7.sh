#!/bin/bash
MODE=$1 # whether to install using sudo or not
set -e
export DEBIAN_FRONTEND=noninteractive
ARCH=$([[ $(uname -m) == x86_64 ]] && echo x86_64 || echo noarch)
$MODE yum install -y https://packages.endpointdev.com/rhel/7/os/${ARCH}/endpoint-repo.${ARCH}.rpm
$MODE yum groupinstall -y "Development Tools"
$MODE yum install -y wget git valgrind centos-release-scl
$MODE yum install -y devtoolset-10
$MODE scl enable devtoolset-10 bash
$MODE update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc10-gcc 60 --slave /usr/bin/g++ g++ /usr/bin/gcc10-g++
source install_cmake.sh $MODE
