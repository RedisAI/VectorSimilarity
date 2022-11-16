#!/bin/bash
set -e
apt-get update -qq
apt-get install -yqq git wget build-essential valgrind 
source linux_install_cmake.sh
