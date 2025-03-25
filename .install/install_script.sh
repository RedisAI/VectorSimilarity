#!/bin/bash

OS_TYPE=$(uname -s)
ARCH=$(uname -m)

if [[ $OS_TYPE = 'Darwin' ]]
then
    OS='macos'
else
    VERSION=$(grep '^VERSION_ID=' /etc/os-release | sed 's/"//g')
    VERSION=${VERSION#"VERSION_ID="}
    OS_NAME=$(grep '^NAME=' /etc/os-release | sed 's/"//g')
    OS_NAME=${OS_NAME#"NAME="}
    [[ $OS_NAME == 'Rocky Linux' ]] && VERSION=${VERSION%.*} # remove minor version for Rocky Linux
    [[ $OS_NAME == 'Alpine Linux' ]] && VERSION=${VERSION%.*.*} # remove minor and patch version for Alpine Linux
    OS=${OS_NAME,,}_${VERSION}
    OS=$(echo $OS | sed 's/[/ ]/_/g') # replace spaces and slashes with underscores
fi
echo $OS
if [[ $ARCH == 'aarch64' ]]
then
    wget https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_24.10/arm-performance-libraries_24.10_deb_gcc.tar
    tar -xf arm-performance-libraries_24.10_deb_gcc.tar
    sudo ./arm-performance-libraries_24.10_deb/arm-performance-libraries_24.10_deb.sh --accept --install-to armpl
    sudo rm -rf arm-performance-libraries_24.10_deb_gcc.tar
    # install libarmpl
elif [[ $OS == 'macos' ]]
then
    wget https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_24.10/arm-performance-libraries_24.10_macOS.tgz
    tar zxvf arm-performance-libraries_24.10_macOS.tgz
    hdiutil attach armpl_24.10_flang-new_clang_19.dmg
    /Volumes/armpl_24.10_flang-new_clang_19_installer/armpl_24.10_flang-new_clang_19_install.sh -y --install-to armpl
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
source ${OS}.sh $1

# input="install.txt"
# apt-get update
# while IFS= read -r line
# do
#   echo $line
#   $($line)
# done < "$OS"
