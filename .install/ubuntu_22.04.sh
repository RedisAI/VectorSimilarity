#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -yqq git wget build-essential valgrind lcov
source linux_install_cmake.sh
