#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT=$HERE/..
READIES=$ROOT/deps/readies
. $READIES/shibumi/functions

cd $HERE
mkdir -p benchmarks/build
(cd benchmarks/build && cmake .. && make)
(cd benchmarks/build && ./openblasbench) 
(cd benchmarks/build && ./mklbench) 


