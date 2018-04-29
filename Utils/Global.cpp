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
int g_debug_mask
#if DEBUG_OUT
		= DEBUG_LOG_LOW;
#else
		= 0;
#endif

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

//! A random number generator.
MTRand rng(1);

//! A normalized random number generator.
vector<Norm *> rgNormrnd;

/*		simulation vars		*/
uint64_t g_simulationStep = 0;

//const BGFLOAT g_synapseStrengthAdjustmentConstant = 1.0e-8;

/* 		misc constants		*/
const BGFLOAT pi = 3.1415926536;

#ifdef PERFORMANCE_METRICS
float t_gpu_rndGeneration;
float t_gpu_advanceNeurons;
float t_gpu_advanceSynapses;
float t_gpu_calcSummation;
float t_host_adjustSynapses;

void printPerformanceMetrics(const float total_time)
{
    cout << "t_gpu_rndGeneration: " << t_gpu_rndGeneration << " ms (" << t_gpu_rndGeneration / total_time * 100 << "%)" << endl;
    cout << "t_gpu_advanceNeurons: " << t_gpu_advanceNeurons << " ms (" << t_gpu_advanceNeurons / total_time * 100 << "%)" << endl;
    cout << "t_gpu_advanceSynapses: " << t_gpu_advanceSynapses << " ms (" << t_gpu_advanceSynapses / total_time * 100 << "%)" << endl;
    cout << "t_gpu_calcSummation: " << t_gpu_calcSummation << " ms (" << t_gpu_calcSummation / total_time * 100 << "%)" << endl;
    cout << "t_host_adjustSynapses: " << t_host_adjustSynapses << " ms (" << t_host_adjustSynapses / total_time * 100 << "%)" << endl;
}
#endif // PERFORMANCE_METRICS

// TODO comment
const string MATRIX_TYPE = "complete";
// TODO comment
const string MATRIX_INIT = "const";
