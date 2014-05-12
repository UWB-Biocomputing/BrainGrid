/** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
 * @authors Aaron Oziel, Sean Blackbourn 
 *
 * Fumitaka Kawasaki (5/3/14):
 * All functions were completed and working. 
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
#ifdef __CUDACC__
#include "Book.h"
#endif

const BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

class LIFGPUModel : public LIFModel  {

public:
	LIFGPUModel();
	~LIFGPUModel();
 
	void setupSim(const SimulationInfo *sim_info, const AllNeurons &neurons, const AllSynapses &synapses);
	void advance(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo *sim_info);
	void updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, IRecorder* simRecorder);
	void cleanupSim(AllNeurons &neurons, SimulationInfo *sim_info);

	struct InverseMap
	{
		//! The beginning index of the incoming dynamic spiking synapse array.
		int* incomingSynapse_begin;

		//! The number of incoming synapses.
		int* synapseCount;

		//! Pointer to synapse inverse map.
		uint32_t* inverseIndex;	

		InverseMap() : num_neurons(0), num_synapses(0)
		{
			incomingSynapse_begin = NULL;
			synapseCount = NULL;
			inverseIndex = NULL;	
		};

		InverseMap(int neuron_count, int synapse_count) : num_neurons(neuron_count), num_synapses(synapse_count)
		{
			incomingSynapse_begin = new int[neuron_count];
			synapseCount = new int[neuron_count];
			inverseIndex = new uint32_t[synapse_count];
		};

		~InverseMap()
		{
			if (num_neurons != 0) {
				delete[] incomingSynapse_begin;
				delete[] synapseCount;
			}
			if (num_synapses != 0) 
				delete[] inverseIndex;
		}

	private:
		int num_neurons;
		int num_synapses;	
	};

private: 
	/* ------------------*\
	|* # Helper Functions
	\* ------------------*/

	void allocDeviceStruct(const SimulationInfo *sim_info, const AllNeurons &allNeuronsHost, const AllSynapses &allSynapsesHost);
	void allocNeuronDeviceStruct( int count, int max_spikes );
	void deleteNeuronDeviceStruct( int count );
	void copyNeuronHostToDevice( const AllNeurons& allNeuronsHost, int count );
	void copyNeuronDeviceToHost( AllNeurons& allNeuronsHost, int count );

	void allocSynapseDeviceStruct( int num_neurons, int max_synapses );
	void deleteSynapseDeviceStruct( int num_neurons, int max_synapses );
	void copySynapseHostToDevice( const AllSynapses& allSynapsesHost, int num_neurons, int max_synapses );
	void copySynapseDeviceToHost( AllSynapses& allSynapsesHost, int num_neurons, int max_synapses );

	void allocSynapseImap( int count );
	void deleteSynapseImap( );
	void copyInverseMapHostToDevice(InverseMap &inverseMapHost, int neuron_count, int synapse_count);
	void copyDeviceSynapseCountsToHost(AllSynapses &allSynapsesHost, int neuron_count);
	void copyDeviceSynapseSumCoordToHost(AllSynapses &allSynapsesHost, int neuron_count, int max_synapses);
	void createSynapseImap( AllSynapses &synapses, const SimulationInfo* sim_info );

	// # Load Memory
	// -------------

	// # Create All Neurons
	// --------------------

	// # Advance Network/Model
	// -----------------------

	// # Update Connections
	// --------------------

	// TODO
	void updateHistory(const int currentStep, BGFLOAT epochDuration, AllNeurons &neuron, const SimulationInfo *sim_infos, IRecorder* simRecorder);
	// TODO
	void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);
	// TODO
	void copyDeviceSpikeHistoryToHost(AllNeurons &allNeuronsHost, const SimulationInfo *sim_info);
	//
	void copyDeviceSpikeCountsToHost(AllNeurons &allNeuronsHost, int numNeurons);
	// TODO
	void clearSpikeCounts(int numNeurons);

	// TODO
	void eraseSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
	// TODO
	void addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, BGFLOAT deltaT);
	// TODO
	void createSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, BGFLOAT* sp, BGFLOAT deltaT, synapseType type);

	/*----------------------------------------------*\
	|  Generic Functions for handling synapse types
	\*----------------------------------------------*/


	/*----------------------------------------------*\
	|  Member variables
	\*----------------------------------------------*/

	//! Pointer to device random noise array.
	float* randNoise_d;

	//! Neuron structure in device memory.
	AllNeurons* allNeuronsDevice;

	//! Synapse structures in device memory.
	AllSynapses* allSynapsesDevice;

	//! Pointer to device inverse map.
	InverseMap* inverseMapDevice;
};
