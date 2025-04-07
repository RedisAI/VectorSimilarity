#!/bin/bash
version=2025.1.0.803
uuid=dc93af13-2b3f-40c3-a41b-2bc05a707a80
prefix=/opt/intel/oneapi
processor=$(uname -m)
OS_TYPE=$(uname -s)
MODE=$1 # whether to install using sudo or not

if [[ $OS_TYPE = 'Linux' ]]
then
    if [[ $processor = 'x86_64' ]]
    then
        filename=intel-onemkl-${version}_offline.sh
        wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${uuid}/${filename}
        chmod u+x ./${filename}
        $MODE ./${filename} -a --silent --eula accept --install-dir ${prefix}
        source ${prefix}/setvars.sh
    fi
fi
