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
(cd benchmarks/build && ./native_avx512_bf_bench) 
(cd benchmarks/build && ./native_avx2_bf_bench) 

