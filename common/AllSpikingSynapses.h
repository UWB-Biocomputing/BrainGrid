/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSpikingSynapses AllSpikingSynapses.h "AllSpikingSynapses.h"
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

#include "AllSynapses.h"

class AllSpikingSynapses : public AllSynapses
{
    public:
        AllSpikingSynapses();
        AllSpikingSynapses(const int num_neurons, const int max_synapses);
        virtual ~AllSpikingSynapses();

        virtual void setupSynapses(const int num_neurons, const int max_synapses);
        virtual void setupSynapses(SimulationInfo *sim_info);
        virtual void cleanupSynapses();
        virtual void resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT) = 0;
        void initSpikeQueue(const uint32_t iSyn);
        virtual bool allowBackPropagation();

    protected:
        bool updateDecay(const uint32_t iSyn, const BGFLOAT deltaT);
        virtual void readSynapse(istream &input, const uint32_t iSyn);
        virtual void writeSynapse(ostream& output, const uint32_t iSyn) const;

    public:
#if defined(USE_GPU)
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) = 0;
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) = 0;
        virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice ) = 0;
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) = 0;
        virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info) = 0;
        virtual void copyDeviceSynapseSumCoordToHost(void* allSynapsesDevice, const SimulationInfo *sim_info) = 0;
        // Update the state of all synapses for a time step
        virtual void advanceSynapses(AllSynapses* allSynapsesDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info) = 0;
        virtual void getFpCreateSynapse(unsigned long long& fpCreateSynapse_h) = 0;
        virtual void getFpPrePostSpikeHit(unsigned long long& fpPreSpikeHit_h, unsigned long long& fpPostSpikeHit_h);
#else
        // Update the state of all synapses for a time step
        virtual void advanceSynapse(const uint32_t iSyn, const BGFLOAT deltaT) = 0;
        virtual void preSpikeHit(const uint32_t iSyn);
        virtual void postSpikeHit(const uint32_t iSyn);
        virtual void createSynapse(const uint32_t iSyn, Coordinate source, Coordinate dest, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type) = 0;

    protected:
        bool isSpikeQueue(const uint32_t iSyn);
#endif
    public:
 
        /*! The time of the last spike.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed & Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
     	 *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified  
         */
        uint64_t *lastSpike;

        /*! The decay for the psr.
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::updateDecay() --- Modified
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Modified
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         */
        BGFLOAT *decay;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::updateDecay() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        BGFLOAT *tau;

        /*! The synaptic transmission delay, descretized into time steps.
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::initSpikeQueue() --- Accessed
         *  - SingleThreadedSpikingModel::preSpikeHit() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         */
        int *total_delay;

#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )
        /*! Pointer to the delayed queue
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::initSpikeQueue() --- Initialized
         *  - SingleThreadedSpikingModel::preSpikeHit() --- Accessed
         *  - SingleThreadedSpikingModel::isSpikeQueue() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized  
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed
         */
        uint32_t *delayQueue;

        /*! The index indicating the current time slot in the delayed queue
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::initSpikeQueue() --- Initialized
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::preSpikeHit() --- Accessed
         *  - SingleThreadedSpikingModel::isSpikeQueue() --- Accessed & Modified
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  
         *  Note: This variable is used in GpuSim_struct.cu but I am not sure 
         *  if it is actually from a synapse. Will need a little help here. -Aaron
         *  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
         */
        int *delayIdx;

        /*! Length of the delayed queue
         *  
         *  Usage: GLOBAL CONSTANT
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::initSpikeQueue() --- Initialized
         *  - SingleThreadedSpikingModel::preSpikeHit() --- Accessed
         *  - SingleThreadedSpikingModel::isSpikeQueue() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialzied
         */
        int *ldelayQueue;
};
