#!/bin/bash

brew update
brew install make
brew install coreutils
source install_cmake.sh

echo "GCC version:"
gcc --version
