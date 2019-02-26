/**
 *      @file IAllNeurons.h
 *
 *      @brief An interface for neuron classes.
 */

#pragma once

using namespace std;

#include "Layout.h"
#include "ClusterInfo.h"

class IAllSynapses;
class SynapseIndexMap;
class IAllNeuronsProps;

class IAllNeurons
{
    public:
        CUDA_CALLABLE virtual ~IAllNeurons() {}

        /**
         *  Assignment operator: copy neurons parameters.
         *
         *  @param  r_neurons  Neurons class object to copy from.
         */
        virtual IAllNeurons &operator=(const IAllNeurons &r_neurons) = 0;

        /**
         *  Create and setup neurons properties.
         */
        virtual void createNeuronsProps() = 0;

        /**
         *  Setup the internal structure of the class. 
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeurons(SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;

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
         *  @param  clr_info    ClusterInfo class to read information from.
         */
        virtual void createAllNeurons(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info) = 0;

        /**
         *  Reads and sets the data for all neurons from input stream.
         *
         *  @param  input       istream to read from.
         *  @param  clr_info    used as a reference to set info for neuronss.
         */
        virtual void deserialize(istream &input, const ClusterInfo *clr_info) = 0;

        /**
         *  Writes out the data in all neurons to output stream.
         *
         *  @param  output      stream to write out to.
         *  @param  clr_info    used as a reference to set info for neuronss.
         */
        virtual void serialize(ostream& output, const ClusterInfo *clr_info) const = 0;

#if defined(USE_GPU)
    public:
        /**
         *  Set neurons properties.
         *
         *  @param  pAllNeuronsProps  Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void setNeuronsProps(void *pAllNeuronsProps) = 0;

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
         *  @param  clr_info               ClusterInfo to refer from.
         *  @param  iStepOffset            Offset from the current simulation step.
         *  @param  neuronsDevice          Pointer to the Neurons object in device memory.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset, IAllNeurons* neuronsDevice) = 0;

        /**
         *  Set some parameters used for advanceNeuronsDevice.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         */
        virtual void setAdvanceNeuronsDeviceParams(IAllSynapses &synapses) = 0;

        /**
         *  Create an AllNeurons class object in device
         *
         *  @param pAllNeurons_d       Device memory address to save the pointer of created AllNeurons object.
         *  @param pAllNeuronsProps_d  Pointer to the neurons properties in device memory.
         */
        virtual void createAllNeuronsInDevice(IAllNeurons** pAllNeurons_d, IAllNeuronsProps *pAllNeuronsProps_d) = 0;

        /**
         * Delete an AllNeurons class object in device
         *
         * @param pAllNeurons_d    Pointer to the AllNeurons object to be deleted in device.
         */
        virtual void deleteAllNeuronsInDevice(IAllNeurons* pAllNeurons_d) = 0;

#else // !defined(USE_GPU)
    public:
        /**
         *  Update internal state of the indexed Neuron (called by every simulation step).
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses         The Synapse list to search from.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
         *  @param  clr_info         ClusterInfo class to read information from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap, const ClusterInfo *clr_info, int iStepOffset) = 0;
#endif // defined(USE_GPU)
};
