#!/bin/bash

cd ../
ROOT=$(pwd)

cd $ROOT/src

cmake -DCMAKE_BUILD_TYPE=Debug .
make