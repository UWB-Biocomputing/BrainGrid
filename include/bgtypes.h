#ifndef __BGTYPES_H_
#define __BGTYPES_H_

#pragma once

// This type is used to measure the difference between 
// IEEE Standard 754 single and double-precision floating 
// point values.
//
// Single-precision (float) calculations are fast, but only
// 23 bits are available to store the decimal.
//
// Double-precision (double) calculations are an order of magnitude
// slower, but 52 bits are available to store the decimal.
//
// We'd like to avoid doubles, if the simulation output doesn't suffer.
//

// For floats, uncomment the following two lines and comment DOUBLEPRECISION and the other #define
//#define FLOAT float

#ifdef _WIN32
// We're stuck with single precision, as defined in windows.h/windef.h
#define SINGLEPRECISION
//#define FLOAT float
#include <Windows.h>
typedef unsigned long long int uint64_t;	//included in inttypes.h, which is not available in WIN32//included in inttypes.h, which is not available in WIN32
typedef unsigned int       uint32_t; // same deal as above
#else
#define DOUBLEPRECISION
#define FLOAT double

// solution to get rid of typedef redefine errors on different platforms
#ifdef TARGET_OS_MAC
  
#elif defined __linux__
	typedef FLOAT* PFLOAT;
#elif defined _WIN32 || defined _WIN64
	#define SINGLEPRECISION
	typedef __int32 int32_t;
	typedef unsigned __int32 uint32_t;
#else
#error "unknown platform"
#endif
#endif // _WIN32
#endif // __BGTYPES_H_
