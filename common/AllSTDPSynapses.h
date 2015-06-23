/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSTDPSynapses AllSTDPSynapses.h "AllSTDPSynapses.h"
 * @brief A container of all synapse data
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllDSSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllDSSynapsesDevice structure.
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

#include "AllSpikingSynapses.h"

class AllSTDPSynapses : public AllSpikingSynapses
{
    public:
        AllSTDPSynapses();
        AllSTDPSynapses(const int num_neurons, const int max_synapses);
        virtual ~AllSTDPSynapses();
 
        virtual void setupSynapses(SimulationInfo *sim_info);
        virtual void setupSynapses(const int num_neurons, const int max_synapses);
        virtual void cleanupSynapses();
        virtual void resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT);
        virtual bool allowBackPropagation();

    protected:
        virtual void readSynapse(istream &input, const uint32_t iSyn);
        virtual void writeSynapse(ostream& output, const uint32_t iSyn) const;
        virtual void initSpikeQueue(const uint32_t iSyn);

    public:
#if defined(USE_GPU)
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info );
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );
        virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice );
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info );
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );
        virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info );
        virtual void copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info);
        virtual void copyDeviceSynapseSumCoordToHost(void* allSynapsesDevice, const SimulationInfo *sim_info);
        // Update the state of all synapses for a time step
        virtual void advanceSynapses(AllSynapses* allSynapsesDevice, AllNeurons* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info);
        virtual void getFpCreateSynapse(unsigned long long& fpCreateSynapse_h);
        virtual void getFpPostSpikeHit(unsigned long long& fpPostSpikeHit_h);

    protected:
        virtual void allocDeviceStruct( AllSTDPSynapses &allSynapses, int num_neurons, int maxSynapsesPerNeuron );
        virtual void deleteDeviceStruct( AllSTDPSynapses& allSynapses );
        virtual void copyHostToDevice( void* allSynapsesDevice, AllSTDPSynapses& allSynapses, int num_neurons, int maxSynapsesPerNeuron );
        virtual void copyDeviceToHost( AllSTDPSynapses& allSynapses, const SimulationInfo *sim_info );

    public:
#else
        // Update the state of synapse for a time step
        virtual void advanceSynapse(const uint32_t iSyn, const SimulationInfo *sim_info, AllNeurons *neurons);
        virtual void createSynapse(const uint32_t iSyn, Coordinate source, Coordinate dest, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type);
        virtual void postSpikeHit(const uint32_t iSyn);

    protected:
        bool isSpikeQueuePost(const uint32_t iSyn);

    private:
        void stdpLearning(const uint32_t iSyn,double delta, double epost, double epre);

#endif
    public:

        // dynamic synapse vars...........
        int *total_delayPost;

        uint32_t *delayQueuePost;

        int *delayIdxPost;

        int *ldelayQueuePost;

        BGFLOAT *tauspost;

        BGFLOAT *tauspre;

        BGFLOAT *taupos;

        BGFLOAT *tauneg;

        BGFLOAT *STDPgap;

        BGFLOAT *Wex;

        BGFLOAT *Aneg;

        BGFLOAT *Apos;

        BGFLOAT *mupos;

        BGFLOAT *muneg;
};

#if defined(__CUDACC__)
extern __global__ void getFpCreateSynapseDevice(void (**fpCreateSynapse_d)(AllSTDPSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType));

extern __global__ void advanceSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapses* allSynapsesDevice, void (*fpChangePSR)(AllSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT), AllSpikingNeurons* allNeuronsDevice, int max_spikes, int width );
    
extern __device__ void stdpLearningDevice(AllSTDPSynapses* allSynapsesDevice, const uint32_t iSyn, double delta, double epost, double epre);
    
extern __device__ bool isSpikeQueueDevice(AllSpikingSynapses* allSynapsesDevice, uint32_t iSyn);
extern __device__ bool isSpikeQueuePostDevice(AllSTDPSynapses* allSynapsesDevice, uint32_t iSyn);
    
extern __device__ uint64_t getSpikeHistoryDevice(AllSpikingNeurons* allNeuronsDevice, int index, int offIndex, int max_spikes);

extern __device__ void createSynapse(AllSTDPSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type);

extern __global__ void getFpPostSpikeHitDevice(void (**fpPostSpikeHit_d)(const uint32_t, AllSTDPSynapses*));
        
extern __device__ void postSpikeHitDevice( const uint32_t iSyn, AllSTDPSynapses* allSynapsesDevice );
#endif
