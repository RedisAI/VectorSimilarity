#!/bin/bash
apt-get update
apt-get upgrade -y
apt-get dist-upgrade -y
apt install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt update
apt-get install -y git wget make valgrind gcc-9 g++-9
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 
source linux_install_cmake.sh
