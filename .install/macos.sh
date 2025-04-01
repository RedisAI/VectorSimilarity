#!/bin/bash

# Without pinning cmake, it will install the latest version(>= 4.0)
# This leads to deps/hiredis failing to compile
# For now we went with pinning cmake to 3.31.6 which is the version that is exists in the current mac OS docker image we use
brew pin cmake

brew update
brew install make
source install_cmake.sh
