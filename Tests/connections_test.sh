#!/bin/bash

echo "TESTING COMPILATION......"

g++ -o test ../TinyXPath/*.cpp ../Core/ParameterManager.cpp ../Core/SimulationInfo.cpp ../Connections/*.cpp  ../Neurons/*.cpp ../Layouts/*.cpp ../Synapses/*.cpp ../Connections/*.cpp ../Matrix/*Matrix.cpp tests_connectionsCreate.cpp -g -DTIXML_USE_STL -I../Utils -I../RNG -I../Matrix -I../TinyXPath -I../Core -I../Neurons -I../Layouts -I../Synapses -I../Connections -I../Recorders -g -std=c++11 -lboost_regex

if [[ $? != 0 ]]; then
    echo -e "\nCompilation unsuccessful; exiting."
    exit 0
fi

./test
rm test
