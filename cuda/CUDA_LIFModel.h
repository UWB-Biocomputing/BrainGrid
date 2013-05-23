/**
 * @brief A leaky-integrate-and-fire (I&F) neural network model for GPU CUDA
 *
 * @class LIFModel LIFModel.h "LIFModel.h"
 *
 * Implements both neuron and synapse behaviour.
 *
 * A standard leaky-integrate-and-fire neuron model is implemented
 * where the membrane potential \f$V_m\f$ of a neuron is given by
 * \f[
 *   \tau_m \frac{d V_m}{dt} = -(V_m-V_{resting}) + R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise})
 * \f]
 * where \f$\tau_m=C_m\cdot R_m\f$ is the membrane time constant,
 * \f$R_m\f$ is the membrane resistance, \f$I_{syn}(t)\f$ is the
 * current supplied by the synapses, \f$I_{inject}\f$ is a
 * non-specific background current and \f$I_{noise}\f$ is a
 * Gaussian random variable with zero mean and a given variance
 * noise.
 *
 * At time \f$t=0\f$ \f$V_m\f$ is set to \f$V_{init}\f$. If
 * \f$V_m\f$ exceeds the threshold voltage \f$V_{thresh}\f$ it is
 * reset to \f$V_{reset}\f$ and hold there for the length
 * \f$T_{refract}\f$ of the absolute refractory period.
 *
 * The exponential Euler method is used for numerical integration.
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 *
 * @authors Derek McLean
 *          Paul Bunn - GPULifModel derived from LIFModel and GPUSim (Auth: Fumitaka Kawasaki,)
 */
#pragma once
#ifndef _CUDA_LIFMODEL_H_
#define _CUDA_LIFMODEL_H_

#include "LIFModel.h"
#include "Coordinate.h"
#include "LIFNeuron_struct.h"
#include "LifSynapse_struct.h"
#include "DelayIdx.h"

#include <vector>
#include <iostream>

using namespace std;

/**
 * Implementation of Model for the Leaky-Integrate-and-Fire model.
 */
class CUDA_LIFModel  : public LIFModel
{

    public:
        CUDA_LIFModel();
        virtual ~CUDA_LIFModel();

        /*
         * Declarations of concrete implementations of Model interface for an Leaky-Integrate-and-Fire
         * model.
         *
         * @see Model.h
         */

		// Only deviations from LIFModel are defined

		void advance(AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info);

    protected:

        /* -----------------------------------------------------------------------------------------
         * # Helper Functions
         * ------------------
         */

        // # Read Parameters
        // -----------------

		// NOTE: ALL functions of LIFModel::TiXmlVisitor must be declared to avoid method hiding
        // Parse an element for parameter values.
        // Required by TiXmlVisitor, which is used by #readParameters
		bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute) { return LIFModel::VisitEnter(element, firstAttribute); };
		/// Visit a document.
		bool VisitEnter( const TiXmlDocument& doc )	{ return LIFModel::VisitEnter(doc); }
		/// Visit a document.
		bool VisitExit( const TiXmlDocument& doc )	{ return LIFModel::VisitExit(doc); }
		/// Visit an element.
		bool VisitExit( const TiXmlElement& element )			{ return LIFModel::VisitExit(element); }
		/// Visit a declaration
		bool Visit( const TiXmlDeclaration& declaration )		{ return LIFModel::Visit(declaration); }
		/// Visit a text node
		bool Visit( const TiXmlText& text )						{ return LIFModel::Visit(text); }
		/// Visit a comment node
		bool Visit( const TiXmlComment& comment )				{ return LIFModel::Visit(comment); }
		/// Visit an unknown node
		bool Visit( const TiXmlUnknown& unknown )				{ return LIFModel::Visit(unknown); }

		bool initializeModel(SimulationInfo *sim_info, AllNeurons& neurons, AllSynapses& synapses);
		void updateWeights(const uint32_t num_neurons, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info);
	private:
#ifdef STORE_SPIKEHISTORY
		//! pointer to an array to keep spike history for one activity epoch
		uint64_t* spikeArray;
#endif // STORE_SPIKEHISTORY
		void dataToCStructs(SimulationInfo *psi, AllNeurons& neurons, AllSynapses& synapses, LifNeuron_struct &neuron_st, LifSynapse_struct &synapse_st);
		void readSpikesFromDevice(uint32_t neuron_count, uint32_t *spikecounts);
		void clearSpikesFromDevice(uint32_t neuron_count);
};

#ifdef _CUDA_LIFModel
//
// Following declarations are only used when compiling CUDA_LIFModel.cu
//

//Forward Declarations
extern "C" {
//! Perform updating neurons and synapses for one activity epoch.
void advanceGPU( 
		SimulationInfo *psi,
		AllNeurons& neurons,
		AllSynapses& synapses, 
		uint32_t maxSynapses
#ifdef STORE_SPIKEHISTORY
		,
		uint64_t* spikeArray,
		uint32_t maxSpikes
#endif // STORE_SPIKEHISTORY
		 );

//! Allocate GPU device memory and copy data from host memory.
void allocDeviceStruct(SimulationInfo * psi, 
		LifNeuron_struct& neuron_st, 
		LifSynapse_struct& synapse_st,
		AllNeurons& neurons,
		AllSynapses& synapses,
#ifdef STORE_SPIKEHISTORY
        uint32_t maxSynapses,
		uint32_t maxSpikes 
#else
        uint32_t maxSynapses
#endif // STORE_SPIKEHISTORY
		);

void copySynapseDeviceToHost( LifSynapse_struct& synapse_h, uint32_t count );

void copyNeuronDeviceToHost( LifNeuron_struct& neuron_h, uint32_t count );

//! Deallocate device memory.
void deleteDeviceStruct( );

//! Create synapse inverse map.
void createSynapseImap(SimulationInfo * psi, uint32_t maxSynapses );

//! Generate random number (normal distribution)
void normalMTGPU(float * randNoise_d);
}

#ifdef STORE_SPIKEHISTORY
//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( uint32_t n, uint64_t* spikeHistory_d, uint64_t simulationStep, uint32_t maxSpikes, uint32_t delayIdx, uint32_t maxSynapses );
#else
//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( uint32_t n, uint64_t simulationStep, uint32_t delayIdx, uint32_t maxSynapses );
#endif // STORE_SPIKEHISTORY
//! Perform updating synapses for one time step.
__global__ void advanceSynapsesDevice( uint32_t n, uint32_t width, uint64_t simulationStep, uint32_t bmask );

//! Calculate neuron/synapse offsets.
__global__ void calcOffsets( uint32_t n, FLOAT* summationPoint_d, uint32_t width, float* randNoise_d );

//! Calculate summation point.
__global__ void calcSummationMap( uint32_t n, uint32_t* inverseMap );

//! Update the network.
__global__ void updateNetworkDevice( FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, uint32_t n, uint32_t width, FLOAT deltaT, FLOAT* W_d, uint32_t maxSynapses );

//! Add a synapse to the network.
__device__ void addSynapse( FLOAT W_new, FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, uint32_t neuron_i, uint32_t source_x, uint32_t source_y, uint32_t dest_x, uint32_t dest_y, uint32_t width, FLOAT deltaT, uint32_t maxSynapses );

//! Create a synapse.
__device__ void createSynapse( uint32_t syn_i, uint32_t source_x, uint32_t source_y, uint32_t dest_x, uint32_t dest_y, FLOAT* sp, FLOAT deltaT, synapseType type );

//! Remove a synapse from the network.
__device__ void removeSynapse( uint32_t neuron_i, uint32_t syn_i );

//! Get the type of synapse.
__device__ synapseType synType( neuronType* rgNeuronTypeMap_d, uint32_t ax, uint32_t ay, uint32_t bx, uint32_t by, uint32_t width );

//! Get the type of synapse (excitatory or inhibitory)
__device__ int32_t synSign( synapseType t );

#ifdef PERFORMANCE_METRICS
//! Calculate effective bandwidth in GB/s.
float getEffectiveBandwidth( uint64_t count, uint32_t Br, uint32_t Bw, float time );
#endif // PERFORMANCE_METRICS

//! Delayed queue index - global to all synapses.
DelayIdx delayIdx;

//! Synapse constant (U)stored in device constant memory.
__constant__ FLOAT synapse_U_d[4] = { 0.32, 0.25, 0.05, 0.5 };	// II, IE, EI, EE

//! Synapse constant(D) stored in device constant memory.
__constant__ FLOAT synapse_D_d[4] = { 0.144, 0.7, 0.125, 1.1 };	// II, IE, EI, EE

//! Synapse constant(F) stored in device constant memory.
__constant__ FLOAT synapse_F_d[4] = { 0.06, 0.02, 1.2, 0.05 };	// II, IE, EI, EE

//! Neuron structure in device constant memory.
__constant__ LifNeuron_struct neuron_st_d[1];

//! Synapse structures in device constant memory.
__constant__ LifSynapse_struct synapse_st_d[1];

__constant__ FLOAT g_synapseStrengthAdjustmentConstant_d = 1.0e-8;



#ifdef STORE_SPIKEHISTORY
//! Pointer to device spike history array.
uint64_t* spikeHistory_d = NULL;	
size_t spikeHistory_d_size = 0;
#endif // STORE_SPIKEHISTORY

//! Pointer to device summation point.
FLOAT* summationPoint_d = NULL;	

//! Pointer to device random noise array.
float* randNoise_d = NULL;	

//! Pointer to device inverse map.
uint32_t* inverseMap_d = NULL;	

//! Pointer to neuron type map.
neuronType* rgNeuronTypeMap_d = NULL;
#endif//_CUDA_LIFModel


#endif
