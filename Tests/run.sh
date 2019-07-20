#!/bin/bash

echo "TESTING COMPILATION......"

# g++ -o test ../TinyXPath/*.cpp ../Core/ParameterManager.cpp tests_ParameterManager.cpp -g
g++ -o test ../TinyXPath/*.cpp ../Core/ParameterManager.cpp tests_ParameterManager.cpp -g -DTIXML_USE_STL -I../Utils -I../RNG -I../Matrix -I../TinyXPath

if [[ $? != 0 ]]; then
    echo -e "\nCompilation unsuccessful; exiting."
    exit 0
fi

echo -e "\nTESTING EXECUTION......"
./test

echo -e "\nTESTING WITH VALGRIND......"
valgrind ./test &> valgrind_out.txt
cat valgrind_out.txt | grep -P -e '( running|in use at exit|total heap usage|no leaks are possible)'
rm valgrind_out.txt
rm test
