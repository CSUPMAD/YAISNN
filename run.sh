#!/bin/bash

build_params=(
    -DDEBUG=1
    -O3
    -std=c++14
    -march=native
    -mtune=native
    -fopenmp
    -ffast-math
)


echo "Building..."
#c++ -std=c++14 -fopenmp -O3 -march=native -mtune=native -ffast-math src/NN.cpp
c++ "${build_params[@]}" src/NN.cpp
echo "Done."

echo "Launching..."
date && time ./a.out && date
echo "Done."

#old
#python analysis/scripts/histplot.py

#echo "Analysing Data..."
#./analysis/scripts/run.sh

