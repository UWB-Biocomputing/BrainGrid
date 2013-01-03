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

#ifdef USE_OMP
#include "omp.h"
#	define OMP(x) x
#else
#define OMP(x)
#endif

#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <vector>
#ifdef _WIN32	//needs to be before #include "bgtypes.h" or the #define FLOAT will cause problems
#include <windows.h>	//warning! windows.h also defines FLOAT
#else
#include <inttypes.h>	//used for uint64_t, unavailable in WIN32
#endif
#include "bgtypes.h"
//PAB #include "RNG/MersenneTwister.h"
#include "RNG/RNG.h" //pab
#include "RNG/norm.h"
#include "Coordinate.h"
#include "DynamicArray.cpp"

using namespace std;

//! If defined, a table with time and each neuron voltage will output to stdout.
//#define DUMP_VOLTAGES

#define DEBUG_OUT
//#define DEBUG_OUT2

#ifdef DEBUG_OUT
#   define DEBUG(x) x
#else
#   define DEBUG(x)
#endif

#ifdef DEBUG_OUT2
#   define DEBUG2(x) x
#else
#   define DEBUG2(x)
#endif

#if defined(USE_GPU)
//! CUDA device ID
extern int g_deviceId;
#endif // USE_GPU

extern const FLOAT g_synapseStrengthAdjustmentConstant;

//! The constant PI.
extern const FLOAT pi;

//! A random number generator.
extern RNG rng;

//! A normalized random number generator.
extern vector<Norm *> rgNormrnd;

//! The current simulation step.
extern uint64_t g_simulationStep;

//! Neuron types.
//!	INH - Inhibitory neuron 
//!	EXC - Excitory neuron
enum neuronType { INH = 1, EXC = 2, NTYPE_UNDEF = 0 };

const int g_nMaxChunkSize = 100;

//! Synapse types.
//!	II - Synapse from inhibitory neuron to inhibitory neuron.
//!	IE - Synapse from inhibitory neuron to excitory neuron.
//!	EI - Synapse from excitory neuron to inhibitory neuron.
//!	EE - Synapse from excitory neuron to excitory neuron.
enum synapseType { II = 0, IE = 1, EI = 2, EE = 3, STYPE_UNDEF = -1 };

//! The default membrane capacitance.
extern const FLOAT DEFAULT_Cm;
//! The default membrane resistance.
extern const FLOAT DEFAULT_Rm;
//! The default resting voltage.
extern const FLOAT DEFAULT_Vrest;
//! The default reset voltage.
extern const FLOAT DEFAULT_Vreset;
//! The default absolute refractory period.
extern const FLOAT DEFAULT_Trefract;
//! The default synaptic noise.
extern const FLOAT DEFAULT_Inoise;
//! The default injected current.
extern const FLOAT DEFAULT_Iinject;
//! The default threshold voltage.  If \f$V_m >= V_{thresh}\f$ then the neuron fires.
extern const FLOAT DEFAULT_Vthresh;
//! The default time step size.
extern const FLOAT DEFAULT_dt;
//! The default absolute refractory period for inhibitory neurons.
extern const FLOAT DEFAULT_InhibTrefract;
//! The default absolute refractory period for excitory neurons.
extern const FLOAT DEFAULT_ExcitTrefract;

//! The default synaptic time constant.
extern const FLOAT DEFAULT_tau;
//! The default synaptic efficiency.
extern const FLOAT DEFAULT_U;
//! The default synaptic efficiency.
extern const FLOAT DEFAULT_delay_weight;

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
#endif // PERFORMANCE_METRICS

#endif
