#! /bin/bash

hostname=$(hostname)

echo "Machine: $hostname"

if [ x"$hostname" == x"cifarm-centos-node.mpi-cbg.de"  ]; then
        echo "CentOS node"
        source /opt/rh/devtoolset-8/enable
fi

mkdir src/config

rm -rf $HOME/openfpm_dependencies/openfpm_devices/BOOST

if [ ! -d $HOME/openfpm_dependencies/openfpm_devices/BOOST ]; then
        if [ x"$hostname" == x"cifarm-mac-node" ]; then
                echo "Compiling for OSX"
                ./install_BOOST.sh $HOME/openfpm_dependencies/openfpm_devices/ 4 darwin
        else
                echo "Compiling for Linux"
                ./install_BOOST.sh $HOME/openfpm_dependencies/openfpm_devices 4 gcc
        fi
fi

sh ./autogen.sh
sh ./configure --with-boost=$HOME/openfpm_dependencies/openfpm_devices/BOOST

make



