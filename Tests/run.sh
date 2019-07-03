#!/bin/bash

echo "TESTING COMPILATION......"

# g++ -o test ../TinyXPath/*.cpp ../Core/ParameterManager.cpp tests_ParameterManager.cpp -g
g++ -o test ../tinyxml/tiny*.cpp ../Core/ParameterManager.cpp tests_ParameterManager.cpp -g -DTIXML_USE_STL -I../Utils -I../RNG -I../Matrix -I../tinyxml

if [[ $? != 0 ]]; then
    echo -e "\nCompilation unsuccessful; exiting."
    exit 0
fi

./test