#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODE=$1 # whether to install using sudo or not


$MODE dnf update -y

# Development Tools includes config-manager
$MODE dnf groupinstall "Development Tools" -yqq

# powertools is needed to install epel
$MODE dnf config-manager --set-enabled powertools

# get epel to install gcc11
$MODE dnf install epel-release -yqq

$MODE dnf install -y gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ gcc-toolset-11-libatomic-devel  make valgrind wget git
cp /opt/rh/gcc-toolset-11/enable /etc/profile.d/gcc-toolset-11.sh

$MODE source /opt/rh/gcc-toolset-11/enable

source install_cmake.sh $MODE
