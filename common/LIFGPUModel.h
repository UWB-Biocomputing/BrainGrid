/** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
 * @authors Aaron Oziel, Sean Blackbourn 
 *
 * Aaron Wrote (2/3/14):
 * This file is extremely out of date and will be need to be updated to
 * reflect changes made to the corresponding .cu file. Functions will need
 * to be added/removed where necessary and the LIFModel super class will need
 * to be edited to reflect a more abstract Model that can be used for both
 * single-threaded and GPU implementations. 
\** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **/
#pragma once
#include "LIFModel.h"

class LIFGPUModel : public LIFModel  {

public:
	void advance(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
	void updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
	void cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info);
	void logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, const SimulationInfo &sim_info) const;

private: 
	/* ------------------*\
	|* # Helper Functions
	\* ------------------*/

	// # Load Memory
	// -------------

	// TODO
	bool updateDecay(AllSynapses &synapses, const int neuron_index, const int synapse_index);

	// # Create All Neurons
	// --------------------

	// TODO
	//void updateNeuron(AllNeurons &neurons, int neuron_index);

	// # Advance Network/Model
	// -----------------------

	// Update the state of all neurons for a time step
	void advanceNeurons(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
	// Helper for #advanceNeurons. Updates state of a single neuron.
	// void advanceNeuron(AllNeurons& neurons, const int index);
	// Initiates a firing of a neuron to connected neurons
	void fire(AllNeurons &neurons, const int index) const;

	// Update the state of all synapses for a time step
	void advanceSynapses(const int num_neurons, AllSynapses &synapses);
	// Helper for #advanceSynapses. Updates state of a single synapse.
	// void advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
	// TODO
	bool isSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index);

	// # Update Connections
	// --------------------

	// TODO
	void updateHistory(int currentStep, BGBGFLOAT epochDuration, AllNeurons &neurons);
	// TODO
	void updateFrontiers(const int num_neurons);
	// TODO
	void updateOverlap(BGBGFLOAT num_neurons);
	// TODO
	void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
	// TODO
	void getSpikeCounts(const AllNeurons &neurons, int *spikeCounts);
	// TODO
	void clearSpikeCounts(AllNeurons &neurons);

	// TODO
	void eraseSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
	// TODO
	void addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGBGFLOAT *sum_point, BGBGFLOAT deltaT);
	// TODO
	void createSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, BGBGFLOAT* sp, BGBGFLOAT deltaT, synapseType type);

	/*----------------------------------------------*\
	|  Generic Functions for handling synapse types
	\*----------------------------------------------*/

	// Determines the type of synapse for a synapse at a given location in the network.
	synapseType synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const int width);
	// Determines the type of synapse for a synapse between two neurons.
	synapseType synType(AllNeurons &neurons, const int src_neuron, const int dest_neuron);
	// Determines the direction of the weight for a given synapse type.
	int synSign(const synapseType t);

	/*----------------------------------------------*\
	|  Member variables
	\*----------------------------------------------*/

	#ifdef STORE_SPIKEHISTORY
	//! Pointer to device spike history array.
	uint64_t* spikeHistory_d = NULL;
	#endif // STORE_SPIKEHISTORY

	//! Pointer to device summation point.
	FLOAT* summationPoint_d = NULL;

	//! Pointer to device random noise array.
	float* randNoise_d = NULL;

	//! Pointer to device inverse map.
	uint32_t* inverseMap_d = NULL;

	//! Pointer to neuron type map.
	neuronType* rgNeuronTypeMap_d = NULL;

};
