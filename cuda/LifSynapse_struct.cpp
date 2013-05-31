/**
 ** \file DynamicSpikingSynapse_struct.cpp
 **
 ** \authors oldman
 ** \brief A dynamic spiking synapse (Makram et al (1998))
 **/

#include "LifSynapse_struct.h"
#include "AllSynapses.h"

/**
 * Allocate data members in the DynamicSpikingSynapse_struct.
 * @param synapse
 * @param count
 */
void allocSynapseStruct(LifSynapse_struct& synapse, int count) {
	synapse.inUse			= new bool[count](); // parentheses ensures initialization
	synapse.summationPoint	= new PBGFLOAT[count]();
	synapse.summationCoord	= new Coordinate[count]();
	synapse.synapseCoord 	= new Coordinate[count]();
	synapse.deltaT 			= new TIMEFLOAT[count]();
	synapse.W 				= new BGFLOAT[count]();
	synapse.psr 			= new BGFLOAT[count]();
	synapse.decay 			= new BGFLOAT[count]();
	synapse.total_delay		= new int[count]();
	synapse.type 			= new synapseType[count]();
	synapse.delayQueue		= new uint32_t[count]();
	synapse.ldelayQueue		= new int[count]();
	synapse.tau 			= new BGFLOAT[count]();
	synapse.r 				= new BGFLOAT[count]();
	synapse.u 				= new BGFLOAT[count]();
	synapse.lastSpike 		= new uint64_t[count]();	
}

/**
 * Deallocate data members in the DynamicSpikingSynapse_struct.
 * @param synapse
 */
void deleteSynapseStruct(LifSynapse_struct& synapse) {
	delete[] synapse.inUse;
	delete[] synapse.summationPoint;
	delete[] synapse.summationCoord;
	delete[] synapse.synapseCoord;
	delete[] synapse.deltaT;
	delete[] synapse.W;
	delete[] synapse.psr;
	delete[] synapse.decay;
	delete[] synapse.total_delay;
	delete[] synapse.type;
	delete[] synapse.delayQueue;
	delete[] synapse.ldelayQueue;
	delete[] synapse.tau;
	delete[] synapse.r;
	delete[] synapse.u;
	delete[] synapse.lastSpike;
}

/**
 * Allocate data members (inUse and summationCoord) in the DynamicSpikingSynapse_struct.
 * @param synapse
 * @param count
 */
void allocSynapseSumCoord(LifSynapse_struct& synapse, int count) {
//	First delete any previously-existing structures
	delete[] synapse.inUse;
	delete[] synapse.summationCoord;
	synapse.inUse           = new bool[count]();
	synapse.summationCoord  = new Coordinate[count]();
}

/**
 * Deallocate data members (inUse and summationCoord) in the DynamicSpikingSynapse_struct.
 * @param synapse
 */
void deleteSynapseSumCoord(LifSynapse_struct& synapse) {
	delete[] synapse.inUse;
	delete[] synapse.summationCoord;
	synapse.inUse           = NULL;
	synapse.summationCoord  = NULL;
}

/**
 * Copy a ISynapse into a LifSynapse_struct.
 * @param in
 * @param out
 * @param idx
 */
void copySynapseToStruct(uint32_t num_neurons, AllSynapses &synapses, int neuron, LifSynapse_struct& out, int idx) {
	// copy everything necessary
	out.inUse[idx] 		= true;
	out.W[idx] 			= synapses.W[neuron * num_neurons + idx];
	out.decay[idx] 		= synapses.decay[neuron * num_neurons + idx];
	out.deltaT[idx] 		= synapses.deltaT[neuron * num_neurons + idx];
	out.lastSpike[idx] 	= synapses.lastSpike[neuron * num_neurons + idx];
	out.psr[idx] 			= synapses.psr[neuron * num_neurons + idx];
	out.r[idx] 			= synapses.r[neuron * num_neurons + idx];
	out.summationCoord[idx]= synapses.summationCoord[neuron * num_neurons + idx];
	out.synapseCoord[idx] 	= synapses.synapseCoord[neuron * num_neurons + idx];
	out.summationPoint[idx]= 0;
	out.tau[idx] 			= synapses.tau[neuron * num_neurons + idx];
	out.total_delay[idx] 	= synapses.total_delay[neuron * num_neurons + idx];
	out.u[idx] 			= synapses.u[neuron * num_neurons + idx];
	out.ldelayQueue[idx] 	= synapses.ldelayQueue[neuron * num_neurons + idx];
	out.type[idx] 			= synapses.type[neuron * num_neurons + idx];
	out.delayQueue[idx] 	= synapses.delayQueue[neuron * num_neurons + idx];
}

/**
 * Copy a LifSynapse_struct into a Synapse.
 * @param in
 * @param out
 * @param idx
 */
void copyStructToSynapse(LifSynapse_struct& in, AllSynapses &synapses, int neuron, int idx) {
	// copy everything necessary
	uint32_t num_neurons = synapses.count_neurons;

	synapses.W[neuron * num_neurons + idx]				= in.W[idx];
	synapses.decay[neuron * num_neurons + idx]			= in.decay[idx];
	synapses.deltaT[neuron * num_neurons + idx] 		= in.deltaT[idx];
	synapses.lastSpike[neuron * num_neurons + idx]		= in.lastSpike[idx];
	synapses.psr[neuron * num_neurons + idx]			= in.psr[idx];
	synapses.r[neuron * num_neurons + idx]				= in.r[idx];
	synapses.summationCoord[neuron * num_neurons + idx]	= in.summationCoord[idx];
	synapses.synapseCoord[neuron * num_neurons + idx]	= in.synapseCoord[idx];
	synapses.tau[neuron * num_neurons + idx]			= in.tau[idx];
	synapses.total_delay[neuron * num_neurons + idx]	= in.total_delay[idx];
	synapses.u[neuron * num_neurons + idx]				= in.u[idx];
	synapses.type[neuron * num_neurons + idx]			= in.type[idx];
	synapses.delayQueue[neuron * num_neurons + idx]		= in.delayQueue[idx];
}

#if 0
//PAB:  Not sure if this will be needed or not yet...

/**
 * Copy a synapseArray into a synapseMap.
 * @param synapse
 * @param synapseMap
 * @param numNeurons
 */
void synapseArrayToMap(DynamicSpikingSynapse_struct* synapse_st, vector<ISynapse*> * synapseMap, int numNeurons, int maxSynapses)
{
	// create a synapse
	BGFLOAT sp;
	DynamicSpikingSynapse* syn = new DynamicSpikingSynapse(0, 0, 0, 0, sp, DEFAULT_delay_weight, 0.0001, II);

	for (int neuron_i = 0; neuron_i < numNeurons; neuron_i++)
	{
		synapseMap[neuron_i].clear();
		for (int j = 0; j < maxSynapses; j++)
		{
			if (synapse_st->inUse[neuron_i * maxSynapses + j] == true)
			{
				copyStructToSynapse( synapse_st, syn, neuron_i * maxSynapses + j );
				synapseMap[neuron_i].push_back(syn);
			}
		}
	}
}
#endif//0