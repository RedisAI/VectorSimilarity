#!/bin/bash
VERSION=$(grep '^VERSION_ID' /etc/os-release | sed 's/"//g')
VERSION=${VERSION#"VERSION_ID="}
OS_NAME=$(grep '^NAME' /etc/os-release | sed 's/"//g')
OS_NAME=${OS_NAME#"NAME="}
OS=${OS_NAME,,}_${VERSION}
OS=${OS// /'_'}
echo $OS

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
source ${OS}.sh

# input="install.txt"
# apt-get update
# while IFS= read -r line
# do
#   echo $line
#   $($line)
# done < "$OS"