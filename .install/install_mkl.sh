#!/bin/bash
filename=l_BaseKit_p_2024.1.0.596.sh
uuid=fdc7a2bc-b7a8-47eb-8876-de6201297144
prefix=/opt/intel/oneapi
processor=$(uname -m)
OS_TYPE=$(uname -s)
MODE=$1 # whether to install using sudo or not

if [[ $OS_TYPE = 'Linux' ]]
then
    if [[ $processor = 'x86_64' ]]
    then
        wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${uuid}/${filename}
        chmod u+x ./${filename}
        $MODE ./${filename} -a --action install --silent --eula accept --components intel.oneapi.lin.mkl.devel --install-dir ${prefix}
        #source ${prefix}/setvars.sh
    fi
fi
