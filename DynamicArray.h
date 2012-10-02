/**
 ** \brief  A templated wrapper class for dynamic 2-d arrays.
 **
 ** \class DynamicArray DynamicArray.h "DynamicArray.h"
 **
 ** Creates a 2-d array of <T>.\n
 ** Memory is allocated continguously.\n
 ** The actual memory consumed is
 ** \f[
 ** 	width \cdot height \cdot sizeof(<T>) + height \cdot sizeof(<T>*).
 ** \f]
 ** where the first dimension(x) is a vector of pointers into the starting location
 ** for each element of the second dimension(y).
 **/

/**
 ** \file DynamicArray.h
 **
 ** Header file for DynamicArray.
 **/

//! Dynamically create a 2-d array of type T.
template<typename T>
T **allocate2dArray(int x, int y);

//! Free memory associated with dynamic array.
template<typename T>
void free2dArray(T** array);
