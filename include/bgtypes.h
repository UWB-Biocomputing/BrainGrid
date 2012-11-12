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
//#define SINGLEPRECISION
//#define FLOAT float

#define DOUBLEPRECISION
#define FLOAT double

// solution to get rid of typedef redefine errors on different platforms
#ifdef TARGET_OS_MAC
  
#elif defined __linux__
	typedef FLOAT* PFLOAT;
#elif defined _WIN32 || defined _WIN64
	typedef __int32 int32_t;
	typedef unsigned __int32 uint32_t;
#else
#error "unknown platform"
#endif

#endif // __BGTYPES_H_
