#!/bin/bash

echo "TESTING COMPILATION......"

g++ -o test ../TinyXPath/*.cpp ../Core/ParameterManager.cpp ../Core/SimulationInfo.cpp ../Core/EventQueue.cpp ../Core/SynapseIndexMap.cpp ../Core/InterClustersEventHandler.cpp ../Core/Model.cpp ../Core/Cluster.cpp ../Recorders/Xml*.cpp ../Utils/*.cpp ../RNG/*.cpp ../Neurons/*.cpp ../Layouts/*.cpp ../Synapses/*.cpp ../Connections/*.cpp ../Matrix/*Matrix.cpp tests_neuronsCreate.cpp -g -DTIXML_USE_STL -I../Utils -I../RNG -I../Matrix -I../TinyXPath -I../Core -I../Neurons -I../Layouts -I../Synapses -I../Connections -I../Recorders -I../Inputs -g -std=c++11 -lboost_regex -pthread

if [[ $? != 0 ]]; then
    echo -e "\nCompilation unsuccessful; exiting."
    exit 0
fi

echo "Compiled successfully."
echo -e "\nRUNNING TESTS......"

./test
rm test
