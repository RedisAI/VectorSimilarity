#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT=$HERE/..
READIES=$ROOT/deps/readies
. $READIES/shibumi/functions

cd $HERE
mkdir -p unit/build
(cd unit/build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make)
(cd unit/build &&./test_hnswlib)

