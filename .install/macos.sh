#!/bin/bash

# Without pinning cmake, it will install the latest version(>= 4.0)
# This leads to to an error:
# Compatibility with CMake < 3.5 has been removed from CMake.
# For now we went with pinning cmake to 3.31.6 which is the version that is exists in the current mac OS docker image we use
brew pin cmake
brew update
brew install make
source install_cmake.sh

VERSION=18
brew install llvm@$VERSION
BREW_PREFIX=$(brew --prefi
LLVM="$BREW_PREFIX/opt/llvm@$VERSION/bin"
GNUBIN=$BREW_PREFIX/opt/make/libexec/gnubin
COREUTILS=$BREW_PREFIX/opt/coreutils/libexec/gnubin

# Update both profile files with all tools

# Source the profile update utility
source "$(dirname "$0")/macos_update_profile.sh"

update_profile ~/.zshrc "$LLVM" "$GNUBIN" "$COREUTILS"
update_profile ~/.zshrc "$LLVM" "$GNUBIN" "$COREUTILS"
