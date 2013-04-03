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

#ifdef DEBUG_OUT
//DEBUG("L")
#   define DEBUG(x) x
#else
#   define DEBUG(x)
#endif

#include <sstream>
#include <cassert>
#include <vector>
#ifdef _WIN32	//needs to be before #include "bgtypes.h" or the #define BGFLOAT will cause problems
#include <windows.h>	//warning! windows.h also defines BGFLOAT
typedef unsigned long long int uint64_t;	//included in inttypes.h, which is not available in WIN32
#else
#include <inttypes.h>	//used for uint64_t, unavailable in WIN32
#endif
#include "include/bgtypes.h"
#include "include/norm.h"
#include "Coordinate.h"
#include "matrix/VectorMatrix.h"

using namespace std;

//! If defined, a table with time and each neuron voltage will output to stdout.
//#define DUMP_VOLTAGES

//#define DEBUG_OUT
//#define DEBUG_OUT2

#ifdef DEBUG_OUT2
#   define DEBUG2(x) x
#else
#   define DEBUG2(x)
#endif

#if defined(USE_GPU)
//! CUDA device ID
extern int g_deviceId;
#endif // USE_GPU

//! The constant PI.
extern const BGFLOAT pi;

//! A random number generator.
extern RNG rng;

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
extern const BGFLOAT DEFAULT_Cm;
//! The default membrane resistance.
extern const BGFLOAT DEFAULT_Rm;
//! The default resting voltage.
extern const BGFLOAT DEFAULT_Vrest;
//! The default reset voltage.
extern const BGFLOAT DEFAULT_Vreset;
//! The default absolute refractory period.
extern const BGFLOAT DEFAULT_Trefract;
//! The default synaptic noise.
extern const BGFLOAT DEFAULT_Inoise;
//! The default injected current.
extern const BGFLOAT DEFAULT_Iinject;
//! The default threshold voltage.  If \f$V_m >= V_{thresh}\f$ then the neuron fires.
extern const BGFLOAT DEFAULT_Vthresh;
//! The default time step size.
extern const BGFLOAT DEFAULT_dt; // MODEL INDEPENDENT
//! The default absolute refractory period for inhibitory neurons.
extern const BGFLOAT DEFAULT_InhibTrefract;
//! The default absolute refractory period for excitory neurons.
extern const BGFLOAT DEFAULT_ExcitTrefract;

//! The default synaptic time constant.
extern const BGFLOAT DEFAULT_tau;
//! The default synaptic efficiency.
extern const BGFLOAT DEFAULT_U;
//! The default synaptic efficiency.
extern const BGFLOAT DEFAULT_delay_weight; // WHAT IS THIS?
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
