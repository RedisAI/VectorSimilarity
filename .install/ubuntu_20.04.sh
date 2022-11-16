#!/bin/bash
apt-get update
apt-get install -y wget make clang-format gcc valgrind python3-pip lcov
source linux_install_cmake.sh
