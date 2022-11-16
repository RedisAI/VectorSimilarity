#!/bin/bash
set -e
apt-get update -qq
apt-get install -yqq wget make clang-format gcc valgrind python3-pip lcov
source linux_install_cmake.sh
