#!/bin/bash
wget https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-linux-x86_64.sh 
chmod u+x ./cmake-3.24.3-linux-x86_64.sh
./cmake-3.24.3-linux-x86_64.sh --skip-license --prefix=/usr/local --exclude-subdir
cmake --version
