/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSynapses AllSynapses.h "AllSynapses.h"
 * @brief A container of all synapse data
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllSynapsesDevice structure.
 *
 *  In this file you will find usage statistics for every variable inthe BrainGrid 
 *  project as we find them. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function()
 *  
 *  For Example
 *
 *  Usage:
 *  - LOCAL VARIABLE -- a variable for individual synapse
 *  - LOCAL CONSTANT --  a constant for individual synapse
 *  - GLOBAL VARIABLE -- a variable for all synapses
 *  - GLOBAL CONSTANT -- a constant for all synapses
 *
 *  Class::function(): --- Initialized, Modified OR Accessed
 *
 *  OtherClass::function(): --- Accessed   
 *
 *  Note: All GLOBAL parameters can be scalars. Also some LOCAL CONSTANT can be categorized 
 *  depending on synapse types. 
 */
#pragma once

#include "Global.h"
#include "SimulationInfo.h"

#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class AllNeurons;

class AllSynapses
{
    public:
        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        virtual ~AllSynapses();

        virtual void setupSynapses(const int num_neurons, const int max_synapses);
        virtual void setupSynapses(SimulationInfo *sim_info);
        virtual void cleanupSynapses();
        virtual void resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT);
        void readSynapses(istream& input, AllNeurons &neurons, const SimulationInfo *sim_info);
        void writeSynapses(ostream& output, const SimulationInfo *sim_info);

    protected:
        virtual void readSynapse(istream &input, const uint32_t iSyn);
        virtual void writeSynapse(ostream& output, const uint32_t iSyn) const;
        synapseType synapseOrdinalToType(const int type_ordinal);

    public:
#if defined(USE_GPU)
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) = 0;
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) = 0;
        virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice ) = 0;
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) = 0;
        virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info) = 0;
        virtual void copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice, const SimulationInfo *sim_info) = 0;
        // Update the state of all synapses for a time step
        virtual void advanceSynapses(AllSynapses* allSynapsesDevice, AllNeurons* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info) = 0;
        virtual void getFpCreateSynapse(unsigned long long& fpCreateSynapse_h) = 0;
        virtual void getFpChangePSR(unsigned long long& fpChangePSR_h) = 0;
#else
        // Update the state of all synapses for a time step
        virtual void advanceSynapses(const SimulationInfo *sim_info, AllNeurons *neurons);
        virtual void advanceSynapse(const uint32_t iSyn, const SimulationInfo *sim_info, AllNeurons *neurons) = 0;
        virtual void eraseSynapse(const int neuron_index, const uint32_t iSyn);
#endif
        virtual void addSynapse(uint32_t &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point, const BGFLOAT deltaT);
        virtual void createSynapse(const uint32_t iSyn, int source_index, int dest_index, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type) = 0;
        int synSign(const synapseType type);

        // TODO
        static const BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;
 
        /*! The location of the synapse.
         *  
         *  Usage: NOT USED ANYWHERE
         *  - LIFModel::loadMemory() --- Iniialized
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        int *sourceNeuronIndex;

        /*! The coordinates of the summation point.
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::loadMemory() --- Accessed
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::updateWeights() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - GPUSpikingModel::copyDeviceSynapseSumCoordToHost() --- Accessed
         *  - GpuSim_Struct::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        int *destNeuronIndex;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::updateWeights() --- Modified 
         *  - SingleThreadedSpikingModel::addSynapse() --- Modified
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
         BGFLOAT *W;

        /*! This synapse's summation point's address.
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::loadMemory() --- Iniialized
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::eraseSynapse() --- Modified (= NULL)
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Initialized
         *  - GpuSim_Struct::calcSummationMap() --- Accessed
         *  - GpuSim_Struct::eraseSynapse() --- Modified (= NULL)
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        BGFLOAT **summationPoint;

    	/*! Synapse type
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        synapseType *type;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized (= 0)
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         *  - GpuSim_Struct::calcSummationMap() --- Accessed
         */
        BGFLOAT *psr;

    	/*! The boolean value indicating the entry in the array is in use.
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeurons() --- Accessed
         *  - SingleThreadedSpikingModel::updateWeights() --- Accessed
         *  - SingleThreadedSpikingModel::eraseSynapse() --- Modified
    	 *  - SingleThreadedSpikingModel::addSynapse() --- Accessed
    	 *  - SingleThreadedSpikingModel::createSynapse() --- Modified
         *  - GPUSpikingModel::copyDeviceSynapseSumCoordToHost() --- Accessed
         *  - GPUSpikingModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Modified
         */
        bool *in_use;

        /*! The number of synapses for each neuron.
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::loadMemory() --- Modified
         *  - LIFModel::saveMemory() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeurons() --- Accessed
    	 *  - SingleThreadedSpikingModel::advanceSynapses() --- Accessed
         *  - SingleThreadedSpikingModel::updateWeights() --- Accessed
    	 *  - SingleThreadedSpikingModel::eraseSynapse() --- Modified
    	 *  - SingleThreadedSpikingModel::addSynapse() --- Modified
         *  - GPUSpikingModel::copyDeviceSynapseCountsToHost() --- Accessed
         *  - GPUSpikingModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::eraseSynapse() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Modified
         * 	     
         *  Note: Likely under a different name in GpuSim_struct, see synapse_count. -Aaron
         */
        size_t *synapse_counts;

        /*! The total number of active synapses.
         *
         *  Usage: GLOBAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - GPUSpikingModel::advance() --- Accessed
         *  - GPUSpikingModel::createSynapseImap() --- Modified
         */
        size_t total_synapse_counts;

    	/*! The maximum number of synapses for each neurons.
         *  
         *  Usage: GLOBAL CONSTANT
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - SingleThreadedSpikingModel::addSynapse() --- Accessed
         *  - GPUSpikingModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::addSynapse --- Accessed
         *  - GpuSim_Struct::createSynapse --- Accessed
         */
        size_t maxSynapsesPerNeuron;

    protected:

        /*! The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int count_neurons;
};
