#!/usr/bin/env bash
if [ ! -d "./data" ]; then
    wget "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" --no-check-certificate
    node converter.js
fi



