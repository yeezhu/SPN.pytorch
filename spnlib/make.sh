#!/bin/bash
HOME=$(pwd)
echo "Compiling cuda kernels..."
cd $HOME/spn/src
rm libspn_kernel.cu.o
nvcc -c -o libspn_kernel.cu.o libspn_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
echo "Installing extension..."
cd $HOME
python setup.py clean && python setup.py install
