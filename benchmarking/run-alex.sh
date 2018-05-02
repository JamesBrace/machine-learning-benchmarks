#!/usr/bin/env bash
yarn benchmark -t mnist -e python -b cpu -l true -p macbook -i 10
yarn benchmark -t mnist -e chrome -b cpu -l true -p macbook -i 1
yarn benchmark -t mnist -e firefox -b cpu -l true -p macbook -i 1



