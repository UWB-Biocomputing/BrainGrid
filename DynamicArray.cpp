/**
 ** \file DynamicArray.cpp
 **
 ** \author Allan Ortiz
 **
 ** \brief Class that provides support for dynamic templated 2-d arrays.
 **/

#include "DynamicArray.h"

template<typename T>
T **allocate2dArray(int x, int y) {
	//(step 1) allocate memory for array of elements of column
	T **array = new T*[x];

	//(step 2) allocate memory for array of elements of each row
	T *cur = new T[y * x];

	// Now point the pointers in the right place
	for (int i = 0; i < x; ++i) {
		*(array + i) = cur;
		cur += y;
	}

	return array;
}


template<typename T>
void free2dArray(T** array) {
	delete[] *array; // delete columns
	delete[] array; // delete top row
}
