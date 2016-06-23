/**
 *	@file global.h
 *
 *	@author Allan Ortiz & Cory Mayberry
 *
 *	@brief Header file for global.h
 *
 */
//! Globally available functions and default parameter values.

#ifndef _GLOBAL_H_
#define _GLOBAL_H_

// Debug output is included in both debug/release builds now.
// The Default for debug is "LOW" and "OFF" for Release.

// Mask bit values:
// 0 (1) -- Normal low-level debugging
// 1 (2) -- Medium level debug info
// 2 (4) -- high/detailed level debug info
// 3 (8) -- parser XML logging
// 4 (16) -- Matrix (CompleteMatrix) debugging
// 5 (32)  -- SparseMatrix debugging
// 6 (64) --  VectorMatrix debugging
#define DEBUG_LOG_LOW       1
#define DEBUG_LOG_MID       2
#define DEBUG_LOG_HI        4
#define DEBUG_LOG_PARSER    8
#define DEBUG_LOG_MATRIX    16
#define DEBUG_LOG_SPARSE    32
#define DEBUG_LOG_VECTOR    64
#define DEBUG_LOG_SYNAPSE   128
#define DEBUG(__x) DEBUG_LOW(__x)
#define DEBUG_LOW(__x)  DEBUG_LOG(DEBUG_LOG_LOW, __x)
#define DEBUG_MID(__x)  DEBUG_LOG(DEBUG_LOG_MID, __x)
#define DEBUG_HI(__x)  DEBUG_LOG(DEBUG_LOG_HI, __x)
#define DEBUG_PARSER(__x) DEBUG_LOG(DEBUG_LOG_PARSER, __x)
#define DEBUG_MATRIX(__x) DEBUG_LOG(DEBUG_LOG_MATRIX, __x)
#define DEBUG_SPARSE(__x) DEBUG_LOG(DEBUG_LOG_SPARSE, __x)
#define DEBUG_VECTOR(__x) DEBUG_LOG(DEBUG_LOG_VECTOR, __x)
#define DEBUG_SYNAPSE(__x) DEBUG_LOG(DEBUG_LOG_SYNAPSE, __x)
#ifdef __CUDACC__
extern __constant__ int d_debug_mask[];
#define DEBUG_LOG(__lvl, __x) { if(__lvl & d_debug_mask[0]) { __x } }
#else
#define DEBUG_LOG(__lvl, __x) { if(__lvl & g_debug_mask) { __x } }
#endif

extern int g_debug_mask;

#include <sstream>
#include <cassert>
#include <vector>
#ifdef _WIN32	//needs to be before #include "bgtypes.h" or the #define BGFLOAT will cause problems
#include <windows.h>	//warning! windows.h also defines BGFLOAT
typedef unsigned long long int uint64_t;	//included in inttypes.h, which is not available in WIN32
#else
#include <inttypes.h>	//used for uint64_t, unavailable in WIN32
#endif
#include "BGTypes.h"
#include "Norm.h"
#include "Coordinate.h"
#include "VectorMatrix.h"

using namespace std;

//! If defined, a table with time and each neuron voltage will output to stdout.
//#define DUMP_VOLTAGES

#if defined(USE_GPU)
//! CUDA device ID
extern int g_deviceId;
#endif // USE_GPU

//! The constant PI.
extern const BGFLOAT pi;

//! A random number generator.
extern MTRand rng;

//! A normalized random number generator.
extern vector<Norm *> rgNormrnd;

//! The current simulation step.
extern uint64_t g_simulationStep;

const int g_nMaxChunkSize = 100;

// NETWORK MODEL VARIABLES NMV-BEGIN {
//! Neuron types.
//!	INH - Inhibitory neuron 
//!	EXC - Excitory neuron
enum neuronType { INH = 1, EXC = 2, NTYPE_UNDEF = 0 };

//! Synapse types.
//!	II - Synapse from inhibitory neuron to inhibitory neuron.
//!	IE - Synapse from inhibitory neuron to excitory neuron.
//!	EI - Synapse from excitory neuron to inhibitory neuron.
//!	EE - Synapse from excitory neuron to excitory neuron.
enum synapseType { II = 0, IE = 1, EI = 2, EE = 3, STYPE_UNDEF = -1 };

//! The default membrane capacitance.
#define DEFAULT_Cm		(3e-8)
//! The default membrane resistance.
#define DEFAULT_Rm		(1e6)
//! The default resting voltage.
#define DEFAULT_Vrest		(0.0)
//! The default reset voltage.
#define DEFAULT_Vreset		(-0.06)
//! The default absolute refractory period.
#define DEFAULT_Trefract	(3e-3)
//! The default synaptic noise.
#define DEFAULT_Inoise		(0.0)
//! The default injected current.
#define DEFAULT_Iinject		(0.0)
//! The default threshold voltage.  If \f$V_m >= V_{thresh}\f$ then the neuron fires.
#define DEFAULT_Vthresh		(-0.04)
//! The default time step size.
#define DEFAULT_dt		(1e-4) // MODEL INDEPENDENT
//! The default absolute refractory period for inhibitory neurons.
#define DEFAULT_InhibTrefract	(2.0e-3)
//! The default absolute refractory period for excitory neurons.
#define DEFAULT_ExcitTrefract	(3.0e-3)

//! The default synaptic time constant.
#define DEFAULT_tau		(3e-3)
//! The default synaptic efficiency.
#define DEFAULT_U		(0.4)
//! The default synaptic efficiency.
#define DEFAULT_delay_weight	(0)
// } NMV-END

//! Converts a 1-d index into a coordinate string.
string index2dToString(int i, int width, int height);
//! Converts a 2-d coordinate into a string.
string coordToString(int x, int y);
//! Converts a 3-d coordinate into a string.
string coordToString(int x, int y, int z);
//! Converts a neuronType into a string.
string neuronTypeToString(neuronType t);

#ifdef PERFORMANCE_METRICS
extern float t_gpu_rndGeneration;
extern float t_gpu_advanceNeurons;
extern float t_gpu_advanceSynapses;
extern float t_gpu_calcSummation;
extern float t_host_adjustSynapses;

void printPerformanceMetrics(const float total_time);
#endif // PERFORMANCE_METRICS

#endif

// TODO comment
extern const string MATRIX_TYPE;
// TODO comment
extern const string MATRIX_INIT;
