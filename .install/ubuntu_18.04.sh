#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not

$MODE apt-get update -qq
$MODE apt-get upgrade -yqq
$MODE apt-get dist-upgrade -yqq
$MODE apt install -yqq software-properties-common
$MODE add-apt-repository ppa:ubuntu-toolchain-r/test -y
$MODE apt update
$MODE apt-get install -yqq git wget make valgrind gcc-10 g++-10
$MODE update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 --slave /usr/bin/g++ g++ /usr/bin/g++-10
source install_cmake.sh $MODE
