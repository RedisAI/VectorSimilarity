#!/bin/bash
set -e
apt-get update -qq
apt-get install -yqq git wget gcc-10 g++-10 make valgrind
source linux_install_cmake.sh
