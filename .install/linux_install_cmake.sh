#!/bin/bash
processor=$(uname -p)
if [[ $processor = 'x86_64' ]]
then
    filename=cmake-3.24.3-linux-x86_64.sh
else
    filename=cmake-3.24.3-linux-aarch64.sh
fi

wget https://github.com/Kitware/CMake/releases/download/v3.24.3/${filename}
chmod u+x ./${filename}
./${filename} --skip-license --prefix=/usr/local --exclude-subdir
cmake --version
