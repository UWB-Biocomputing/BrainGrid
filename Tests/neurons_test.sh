#!/bin/bash

echo "TESTING COMPILATION......"

g++ -o test ../TinyXPath/*.cpp ../Core/ParameterManager.cpp tests_neuronsCreate.cpp -g -DTIXML_USE_STL -I../Utils -I../RNG -I../Matrix -I../TinyXPath -I../Core -I../Neurons -I../Layouts -I../Synapses -I../Connections -I../Recorders -g -std=c++11

if [[ $? != 0 ]]; then
    echo -e "\nCompilation unsuccessful; exiting."
    exit 0
fi

./test
rm test
