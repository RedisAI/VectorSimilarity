#!/bin/bash
cd ./module
cmake -DCMAKE_BUILD_TYPE=Debug .
make

cd ../flow
RLTest --module ../module/testmod.so --clear-logs
