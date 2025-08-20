#!/bin/bash
MODE=$1
set -e

$MODE apk update
$MODE apk add --no-cache build-base gcc g++ make wget git valgrind linux-headers cmake

# Force C++17 and prevent the C++20 deprecation from being a hard error
export CXXFLAGS="$CXXFLAGS -std=gnu++17 -Wno-error=deprecated -Wno-error=deprecated-declarations"
export CFLAGS="$CFLAGS -std=gnu++17"

# (Optional) persist for subsequent CI steps/shells
echo 'export CXXFLAGS="$CXXFLAGS -std=gnu++17 -Wno-error=deprecated -Wno-error=deprecated-declarations"' | $MODE tee /etc/profile.d/cpp17.sh >/dev/null
echo 'export CFLAGS="$CFLAGS -std=gnu++17"' | $MODE tee -a /etc/profile.d/cpp17.sh >/dev/null
