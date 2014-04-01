#pragma once

#include "LIFModel.h"

class LIFSingleThreadedModel : public LIFModel {

public:
	//Constructor & Destructor
	LIFSingleThreadedModel();
	~LIFSingleThreadedModel();

	void advance(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo *sim_info);
	void updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);
	void cleanupSim(AllNeurons &neurons, SimulationInfo *sim_info);
	void logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, const SimulationInfo *sim_info) const;

	/* -----------------
	* # Helper Functions
	* ------------------
	*/

	// # Load Memory
	// -------------

	// TODO
	bool updateDecay(AllSynapses &synapses, const int neuron_index, const int synapse_index, const BGFLOAT deltaT);

	// # Create All Neurons
	// --------------------

	// TODO
	void initNeuronConstsFromParamValues(AllNeurons &neurons, int neuron_index, const BGFLOAT deltaT);

	// # Advance Network/Model
	// -----------------------

	// Update the state of all neurons for a time step
	void advanceNeurons(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo *sim_info);
	// Helper for #advanceNeuron. Updates state of a single neuron.
	void advanceNeuron(AllNeurons& neurons, const int index, const BGFLOAT deltaT);
	// Initiates a firing of a neuron to connected neurons
	void fire(AllNeurons &neurons, const int index, const BGFLOAT deltaT) const;
	// TODO
	void preSpikeHit(AllSynapses &synapses, const int neuron_index, const int synapse_index);

	// Update the state of all synapses for a time step
	void advanceSynapses(const int num_neurons, AllSynapses &synapses, const BGFLOAT deltaT);
	// Helper for #advanceSynapses. Updates state of a single synapse.
	void advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, const BGFLOAT deltaT);
	// TODO
	bool isSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index);

	// # Update Connections
	// --------------------

	// TODO
	void updateHistory(int currentStep, BGFLOAT epochDuration, AllNeurons &neurons, const SimulationInfo *sim_info);
	// TODO
	void updateFrontiers(const int num_neurons);
	// TODO
	void updateOverlap(BGFLOAT num_neurons);
	// TODO
	void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);

	// TODO
	void getSpikeCounts(const AllNeurons &neurons, int *spikeCounts, const SimulationInfo *sim_info);
	// TODO
	void clearSpikeCounts(AllNeurons &neurons, const SimulationInfo *sim_info);

	// TODO
	void eraseSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
	// TODO
	void addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, const BGFLOAT deltaT);
	// TODO
	void createSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type);

	// -----------------------------------------------------------------------------------------
	// # Generic Functions for handling synapse types
	// ---------------------------------------------

	// Determines the type of synapse for a synapse at a given location in the network.
	synapseType synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const int width);
	// Determines the type of synapse for a synapse between two neurons.
	synapseType synType(AllNeurons &neurons, const int src_neuron, const int dest_neuron);
	// Determines the direction of the weight for a given synapse type.
	int synSign(const synapseType t);

};




