#!/bin/bash
# $1: path to data-dir
python3 src/preprocess.py -i ETHOS_Binary -d $1
python3 src/split_ETHOS.py -d $1
