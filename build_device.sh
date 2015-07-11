#! /bin/bash

mkdir src/config

sh ./autogen.sh
if [ "$2" == "master" ]
then
 sh ./configure --disable-gpu
else
 sh ./configure
fi

make

./src/mem


