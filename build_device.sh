#! /bin/bash

mkdir src/config

sh ./autogen.sh
if [ "$2" == "master" ]
then
 sh ./configure --disable-gpu
elif [ "$2" == "gin" ]
then
 module load gcc/4.9.2
 module load boost/1.54.0
 sh ./configure --with-boost=/sw/apps/boost/1.54.0/
else
 sh ./configure
fi

make

./src/mem


