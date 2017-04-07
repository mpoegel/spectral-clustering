#! /bin/bash

if [ ! -f data/raw/usps ]; then
  wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2 -P data/raw
  bzip2 -d data/raw/usps.bz2
fi

if [ ! -f data/raw/mnist ]; then
  wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2 -P data/raw
  bzip2 -d data/raw/mnist.scale.bz2
  mv data/raw/mnist.scale data/raw/mnist
fi
