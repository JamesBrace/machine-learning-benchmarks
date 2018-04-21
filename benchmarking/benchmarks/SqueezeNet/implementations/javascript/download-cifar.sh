#!/usr/bin/env bash
if [ ! -d "./data" ]; then
    wget "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" --no-check-certificate
    tar -xf cifar-10-binary.tar.gz
    cd cifar-10-batches-bin
    mv *.bin ../
    cd ..
    rm -r cifar-10-batches-bin
    node converter.js
    rm *.bin
    rm cifar-10-binary.tar.gz
fi



