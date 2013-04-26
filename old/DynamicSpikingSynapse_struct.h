/**
 ** \brief A Dynamic Spiking Synapse structure.
 ** 
 ** \struct DynamicSpikingSynapse_struct DynamicSpikingSynapse_struct.h "DynamicSpikingSynapse_struct.h"
 ** 
 ** \htmlonly  <h3>Model</h3> \endhtmlonly
 ** 
 ** The time varying state \f$x(t)\f$ of the synapse is increased by \f$W\cdot r \cdot u\f$ when a
 ** presynaptic spike hits the synapse and decays exponentially (time constant \f$\tau\f$) otherwise.
 ** \f$u\f$ and \f$r\f$ model the current state of facilitation and depression.\n
 ** A spike causes an exponential decaying postsynaptic response of the form \f$\exp(-t/\tau)\f$.
 ** 
 ** @authors Allan Ortiz & Cory Mayberry
 ** 
 **/
 
/** 
 * \file DynamicSpikingSynapse_struct.h
 * 
 * Header file for DynamicSpikingSynapse_struct
 *
 */

#ifndef DYNAMICSPIKINGSYNAPSE_STRUCT_H_
#define DYNAMICSPIKINGSYNAPSE_STRUCT_H_


#include "DynamicSpikingSynapse.h"
#include <string.h>

// forward declaration
struct DynamicSpikingSynapse_struct;

/**
 * Allocate data members in the DynamicSpikingSynapse_struct.
 * @param synapse
 * @param count
 */
void allocSynapseStruct(DynamicSpikingSynapse_struct* synapse, int count);

/**
 * Deallocate data members in the DynamicSpikingSynapse_struct.
 * @param synapse
 */
void deleteSynapseStruct(DynamicSpikingSynapse_struct* synapse);

/**
 * Allocate data members (inUse and summationCoord) in the DynamicSpikingSynapse_struct.
 * @param synapse
 * @param count
 */
void allocSynapseSumCoord(DynamicSpikingSynapse_struct* synapse, int count);

/**
 * Deallocate data members (inUse and summationCoord) in the DynamicSpikingSynapse_struct.
 * @param synapse
 */
void deleteSynapseSumCoord(DynamicSpikingSynapse_struct* synapse);

/**
 * Copy a ISynapse into a DynamicSpikingSynapse_struct.
 * @param in
 * @param out
 * @param idx
 */
void copySynapseToStruct(ISynapse* in, DynamicSpikingSynapse_struct* out, int idx);

/**
 * Copy a DynamicSpikingSynapse_struct into a ISynapse.
 * @param in
 * @param out
 * @param idx
 */
void copyStructToSynapse(DynamicSpikingSynapse_struct* in, ISynapse* out, int idx);

/**
 * Copy a synapseArray into a synapseMap.
 * @param synapse
 * @param synapseMap
 * @param numNeurons
 */
void synapseArrayToMap(DynamicSpikingSynapse_struct* synapse, vector<ISynapse*>* synapseMap,
		int numNeurons, int maxSynapses);

struct DynamicSpikingSynapse_struct {
	//! This synase is in use.
	bool* inUse;

	//! This synapse's summation point's address.
	PBGFLOAT* summationPoint;

	//! The coordinates of the summation point.
	Coordinate* summationCoord;

	//! The synapse's source location.
	Coordinate* synapseCoord;

	//! The time step size.
	BGFLOAT* deltaT;

	//! The weight (scaling factor, strength, maximal amplitude) of the synapse.
	BGFLOAT* W;
	//! The post-synaptic response is the result of whatever computation is going on in the synapse.
	BGFLOAT* psr;
	//! The decay for the psr.
	BGFLOAT* decay;
	//! The synaptic transmission delay, descretized into time steps.
	int* total_delay;
	//! Synapse type
	synapseType* type;

	//! The delayed queue
	uint32_t* delayQueue;
	//! Length of the delayed queue
	int* ldelayQueue;

	 //! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)];.
	BGFLOAT* tau;

	// dynamic synapse vars...........
	//! The time varying state variable \f$r\f$ for depression.
	BGFLOAT* r;
	//! The time varying state variable \f$u\f$ for facilitation.
	BGFLOAT* u;
	//! The time of the last spike.
	uint64_t* lastSpike;
};

#endif /* DYNAMICSPIKINGSYNAPSE_STRUCT_H_ */
