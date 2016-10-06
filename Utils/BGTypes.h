#ifndef __BGTYPES_H_
#define __BGTYPES_H_

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


// For floats, uncomment the following two lines and comment DOUBLEPRECISION and
// the other #define BGFLOAT; vice-versa for doubles.

#define SINGLEPRECISION
#define BGFLOAT float
//#define DOUBLEPRECISION
//#define BGFLOAT double

typedef BGFLOAT* PBGFLOAT;

// TIMEFLOAT is used by the GPU code and needs to be a double
#define TIMEFLOAT double


// Platform Specific (are the typdef's redundant?)
#ifdef __linux__ 
typedef unsigned int uint32_t;
typedef signed int int32_t;

#elif defined __APPLE__
typedef unsigned int uint32_t;
typedef signed int int32_t;

#elif defined _WIN32 || defined _WIN64
typedef unsigned __int32 uint32_t;		// included in inttypes.h, which is not 
										// available in WIN32
typedef signed __int32 int32_t;
typedef unsigned long long int uint64_t;

#else
#error "unknown platform"
#endif // Platform Specific


// AMP
#ifdef USE_AMP
#define GPU_COMPAT_BOOL uint32_t
#else
#define GPU_COMPAT_BOOL bool
#endif // AMP

// The type for using array indexes (issue #142).
#define BGSIZE uint32_t
//#define BGSIZE uint64_t

#endif // __BGTYPES_H_


