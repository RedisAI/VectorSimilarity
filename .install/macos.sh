#!/bin/bash

# Without pinning cmake, it will install the latest version(>= 4.0)
# This leads to to an error:
# Compatibility with CMake < 3.5 has been removed from CMake.
# For now we went with pinning cmake to 3.31.6 which is the version that is exists in the current mac OS docker image we use
brew pin cmake
brew update
brew install make
brew install coreutils
source install_cmake.sh
