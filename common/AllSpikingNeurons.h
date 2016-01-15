/**
 *      @file AllSpikingNeurons.h
 *
 *      @brief A container of all spiking neuron data
 */

/**
 ** @class AllSpikingNeurons AllSpikingNeurons.h "AllSpikingNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of all spiking neuron data.
 ** This is the base class of all spiking neuron classes.
 **
 ** The class uses a data-centric structure, which utilizes a structure as the containers of
 ** all neuron.
 **
 ** The container holds neuron parameters of all neurons.
 ** Each kind of neuron parameter is stored in a 1D array, of which length
 ** is number of all neurons. Each array of a neuron parameter is pointed by a
 ** corresponding member variable of the neuron parameter in the class.
 ** 
 ** This structure was originally designed for the GPU implementation of the
 ** simulator, and this refactored version of the simulator simply uses that design for
 ** all other implementations as well. This is to simplify transitioning from
 ** single-threaded to multi-threaded.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **/

#pragma once

using namespace std;

#include "Global.h"
#include "SimulationInfo.h"
#include "AllNeurons.h"

class AllSpikingNeurons : public AllNeurons
{
    public:
        /** 
         *  The booleans which track whether the neuron has fired.
         */
        bool *hasFired;

        /** 
         *  The number of spikes since the last growth cycle.
         */
        int *spikeCount;

        /**
         *  Offset of the spike_history buffer.
         */
        int *spikeCountOffset;

        /** 
         *  Step count (history) for each spike fired by each neuron.
         *  The step counts are stored in a buffer for each neuron, and the pointers
         *  to the buffer are stored in a list pointed by spike_history. 
         *  Each buffer is a circular, and offset of top location of the buffer i is
         *  specified by spikeCountOffset[i].
         */
        uint64_t **spike_history;

        AllSpikingNeurons();
        virtual ~AllSpikingNeurons();

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        virtual void setupNeurons(SimulationInfo *sim_info);

        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        virtual void cleanupNeurons(); 

        /**
         *  Returns the number of required parameters to read.
         */
        virtual int numParameters() = 0;

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
        virtual void readNeurons(istream &input, const SimulationInfo *sim_info) = 0;

        /**
         *  Writes out the data in all neurons to output stream.
         *
         *  @param  output      stream to write out to.
         *  @param  sim_info    used as a reference to set info for neuronss.
         */
        virtual void writeNeurons(ostream& output, const SimulationInfo *sim_info) const = 0;

        /**
         *  Clear the spike counts out of all Neurons.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        void clearSpikeCounts(const SimulationInfo *sim_info);

#if defined(USE_GPU)
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
         *  Copy spike history data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;

        /**
         *  Copy spike counts data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;

        /**
         *  Clear the spike counts out of all neurons.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;

    protected:
        /**
         *  Copy spike history data stored in device memory to host.
         *  (Helper function of copyNeuronDeviceSpikeHistoryToHost)
         *
         *  @param  allNeurons        Reference to the allNeurons struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void copyDeviceSpikeHistoryToHost( AllSpikingNeurons& allNeurons, const SimulationInfo *sim_info );

        /**
         *  Copy spike counts data stored in device memory to host.
         *  (Helper function of copyNeuronDeviceSpikeCountsToHost)
         *
         *  @param  allNeurons         Reference to the allNeurons struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void copyDeviceSpikeCountsToHost( AllSpikingNeurons& allNeurons, const SimulationInfo *sim_info );

        /**
         *  Clear the spike counts out of all neurons in device memory.
         *  (helper function of clearNeuronSpikeCounts)
         *
         *  @param  allNeurons         Reference to the allNeurons struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void clearDeviceSpikeCounts( AllSpikingNeurons& allNeurons, const SimulationInfo *sim_info );

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
        virtual void advanceNeurons(AllSynapses &synapses, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice) = 0;

    public:
        /**
         *  Set some parameters used for advanceNeuronsDevice.
         *  Currently we set the two member variables: m_fpPreSpikeHit_h and m_fpPostSpikeHit_h.
         *  These are function pointers for PreSpikeHit and PostSpikeHit device functions
         *  respectively, and these functions are called from advanceNeuronsDevice device
         *  function. We use this scheme because we cannot not use virtual function (Polymorphism)
         *  in device functions.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         */
        virtual void setAdvanceNeuronsDeviceParams(AllSynapses &synapses);

#else // !defined(USE_GPU)
        /**
         *  Update internal state of the indexed Neuron (called by every simulation step).
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses         The Synapse list to search from.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
         */
        virtual void advanceNeurons(AllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap);

        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index            Index of the neuron to update.
         *  @param  sim_info         SimulationInfo class to read information from.
         */
        virtual void advanceNeuron(const int index, const SimulationInfo *sim_info) = 0;

        /**
         *  Initiates a firing of a neuron to connected neurons
         *
         *  @param  index            Index of the neuron to fire.
         *  @param  sim_info         SimulationInfo class to read information from.
         */
        virtual void fire(const int index, const SimulationInfo *sim_info) const;

    public:
        /**
         *  Get the spike history of neuron[index] at the location offIndex.
         *
         *  @param  index            Index of the neuron to get spike history.
         *  @param  offIndex         Offset of the history buffer to get from.
         *  @param  sim_info         SimulationInfo class to read information from.
         */
        uint64_t getSpikeHistory(int index, int offIndex, const SimulationInfo *sim_info);
#endif // defined(USE_GPU)

    protected:
        /**
         *  True if back propagaion is allowed.
         *  (parameters used for advanceNeuronsDevice.)
         */
        bool m_fAllowBackPropagation;

        /**
         *  Pointer to the device function preSpikeHit() function.
         *  (parameters used for advanceNeuronsDevice.)
         */
        unsigned long long m_fpPreSpikeHit_h;

        /**
         *  Pointer o the device function postSpikeHit().
         *  (parameters used for advanceNeuronsDevice.)
         */
        unsigned long long m_fpPostSpikeHit_h;

    private:
        void freeResources();
};
