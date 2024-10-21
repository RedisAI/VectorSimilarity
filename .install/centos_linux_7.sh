#!/bin/bash
MODE=$1 # whether to install using sudo or not
set -e
export DEBIAN_FRONTEND=noninteractive
ARCH=$([[ $(uname -m) == x86_64 ]] && echo x86_64 || echo noarch)

# http://mirror.centos.org/centos/7/ is deprecated, so we have to disable mirrorlists
# and change the baseurl in the repo file to the working mirror (from mirror.centos.org to vault.centos.org)
set_all_baseurls() {
    for file in /etc/yum.repos.d/*.repo; do
        $MODE sed -i 's/^mirrorlist=/#mirrorlist=/g' $file
        $MODE sed -i 's/^#[[:space:]]*baseurl=http:\/\/mirror/baseurl=http:\/\/vault/g' $file
    done
}

set_all_baseurls # set the baseurls to the working mirror before installing basic packages

$MODE yum install -y https://packages.endpointdev.com/rhel/7/os/${ARCH}/endpoint-repo.${ARCH}.rpm
$MODE yum groupinstall -y "Development Tools"
$MODE yum install -y wget git valgrind centos-release-scl
set_all_baseurls # set the baseurls again before installing devtoolset-11 (some new repos were added)
$MODE yum install -y devtoolset-11
$MODE scl enable devtoolset-11 bash
$MODE yum remove -y gcc # remove gcc 4
$MODE update-alternatives --install /usr/bin/gcc gcc /opt/rh/devtoolset-11/root/usr/bin/gcc 60 \
                            --slave /usr/bin/g++ g++ /opt/rh/devtoolset-11/root/usr/bin/g++
source install_cmake.sh $MODE
