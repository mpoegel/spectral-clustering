#! /bin/bash

if [ ! -d "lib/Eigen" ]; then
  mkdir -p lib/Eigen
  wget http://bitbucket.org/eigen/eigen/get/3.3.1.tar.bz2 -P lib
  tar -vxjf lib/3.3.1.tar.bz2 -C lib/Eigen --strip-components=1
  rm -f lib/3.3.1.tar.bz2
fi

if [ ! -d "lib/spectra-0.4.0" ]; then
  wget https://github.com/yixuan/spectra/archive/v0.4.0.zip -P lib
  unzip lib/v0.4.0.zip -d lib/
  rm -f lib/v0.4.0.zip
fi

if [ ! -d "lib/QCustomPlot" ]; then
  wget http://www.qcustomplot.com/release/2.0.0-beta/QCustomPlot.tar.gz -P lib
  mkdir -p lib/QCustomPlot
  tar -vxzf lib/QCustomPlot.tar.gz -C lib/QCustomPlot --strip-components=1
  rm -f lib/QCustomPlot.tar.gz
fi
