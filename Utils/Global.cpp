/*
 *
 *	@file global.cpp
 *
 *	@author Allan Ortiz and Cory Mayberry
 *
 *  @brief Globally available functions/variables and default parameter values.
 *
 */
#include "Global.h"

// Debugging log data and routines
// see "global.h" for bitmask usage of debug outputs

/*
 *  Converts the given index to a string with the indexes of a two-dimensional array.
 *  @param  i   index to be converted.
 *  @param  width   width of the two-dimensional array
 *  @param  height  height of the two-dimensional array
 *  @return string with the converted indexes and square brackets surrounding them.
 */
string index2dToString(int i, int width, int height) {
  stringstream ss;
  ss << "[" << i % width << "][" << i / height << "]";
  return ss.str();
}

/*
 *  Takes the two given coordinates and outputs them with brackets.
 *  @param  x   x coordinate.
 *  @param  y   y coordinate.
 *  @return returns the given coordinates surrounded by square brackets.
 */
string coordToString(int x, int y) {
  stringstream ss;
  ss << "[" << x << "][" << y << "]";
  return ss.str();
}

/*
 *  Takes the three given coordinates and outputs them with brackets.
 *  @param  x   x coordinate.
 *  @param  y   y coordinate.
 *  @param  z   z coordinate.
 *  @return returns the given coordinates surrounded by square brackets.
 */
string coordToString(int x, int y, int z) {
  stringstream ss;
  ss << "[" << x << "][" << y << "][" << z << "]";
  return ss.str();
}

// MODEL INDEPENDENT FUNCTION NMV-BEGIN {
string neuronTypeToString(neuronType t) {
  switch (t) {
  case INH:
    return "INH";
  case EXC:
    return "EXC";
  default:
    cerr << "ERROR->neuronTypeToString() failed, unknown type: " << t << endl;
    assert(false);
    return NULL; // Must return a value -- this will probably cascade to another failure
  }
}
// } NMV-END
#if defined(USE_GPU)
//! CUDA device ID
int g_deviceId = 0;
#endif // USE_GPU

// number of clusters
int g_numClusters = 1;

//! A random number generator.
MTRand rng(1);

/*		simulation vars		*/
uint64_t g_simulationStep = 0;

//const BGFLOAT g_synapseStrengthAdjustmentConstant = 1.0e-8;

/* 		misc constants		*/
const BGFLOAT pi = 3.1415926536;

#ifdef PERFORMANCE_METRICS
// All times in seconds
double t_host_initialization;
double t_host_advance;
double t_host_adjustSynapses;
double t_host_createSynapseImap;
#endif // PERFORMANCE_METRICS 

// TODO comment
const string MATRIX_TYPE = "complete";
// TODO comment
const string MATRIX_INIT = "const";
