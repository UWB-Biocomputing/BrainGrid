/**
 *      @file IAllNeurons.h
 *
 *      @brief An interface for neuron classes.
 */

#pragma once

using namespace std;

#include "SynapseIndexMap.h"
#include "Layout.h"

class IAllSynapses;

class IAllNeurons
{
    public:
        virtual ~IAllNeurons() {}

        /**
         *  Setup the internal structure of the class. 
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        virtual void setupNeurons(SimulationInfo *sim_info) = 0;

        /**
         *  Cleanup the class.
         *  Deallocate memories. 
         */
        virtual void cleanupNeurons() = 0;

        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters() = 0;

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element) = 0;

        /**
         *  Prints out all parameters of the neurons to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const = 0;

        /**
         *  Creates all the Neurons and assigns initial data for them.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         */
        virtual void createAllNeurons(SimulationInfo *sim_info, Layout *layout) = 0;

        /**
         *  Outputs state of the neuron chosen as a string.
         *
         *  @param  i   index of the neuron (in neurons) to output info from.
         *  @return the complete state of the neuron.
         */
        virtual string toString(const int i) const = 0;

        /**
         *  Reads and sets the data for all neurons from input stream.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neuronss.
         */
        virtual void deserialize(istream &input, const SimulationInfo *sim_info) = 0;

        /**
         *  Writes out the data in all neurons to output stream.
         *
         *  @param  output      stream to write out to.
         *  @param  sim_info    used as a reference to set info for neuronss.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info) const = 0;

#if defined(USE_GPU)
    public:
        /**
         *  Allocate GPU memories to store all neurons' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) = 0;

        /**
         *  Delete GPU memories.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;

        /**
         *  Copy all neurons' data from host to device.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;

        /**
         *  Copy all neurons' data from device to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;

        /**
         *  Update the state of all neurons for a time step
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
         *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
         *  @param  sim_info               SimulationInfo to refer from.
         *  @param  randNoise              Reference to the random noise array.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice) = 0;

        /**
         *  Set some parameters used for advanceNeuronsDevice.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         */
        virtual void setAdvanceNeuronsDeviceParams(IAllSynapses &synapses) = 0;
#else // !defined(USE_GPU)
    public:
        /**
         *  Update internal state of the indexed Neuron (called by every simulation step).
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses         The Synapse list to search from.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap) = 0;
#endif // defined(USE_GPU)
};
