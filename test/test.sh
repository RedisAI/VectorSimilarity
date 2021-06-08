#!/bin/bash
mkdir -p module/build
cd ./module/build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

cd ../../flow
RLTest --module ../module/build/testmod.so --clear-logs
