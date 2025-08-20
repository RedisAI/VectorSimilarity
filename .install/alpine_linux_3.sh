#!/bin/bash
MODE=$1 # whether to install using sudo or not
set -e

$MODE apk update

# pin GCC/G++ to 13 (avoid unversioned gcc/g++ from build-base)
$MODE apk add --no-cache make binutils musl-dev gcc-13 g++-13 wget git valgrind linux-headers

$MODE apk add --no-cache cmake
