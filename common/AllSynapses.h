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

#ifndef _ALLSYNAPSES_H_
#define _ALLSYNAPSES_H_

#include "Global.h"
#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

struct AllSynapses
{
    public:
        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        ~AllSynapses();
 
        /*! The coordinates of the summation point.
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::loadMemory() --- Accessed
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::updateWeights() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - LIFGPUModel::copyDeviceSynapseSumCoordToHost() --- Accessed
         *  - GpuSim_Struct::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        Coordinate **summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed
         *  - LIFSingleThreadedModel::updateWeights() --- Modified 
         *  - LIFSingleThreadedModel::addSynapse() --- Modified
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
         BGFLOAT **W;

        /*! This synapse's summation point's address.
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::loadMemory() --- Iniialized
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed
         *  - LIFSingleThreadedModel::eraseSynapse() --- Modified (= NULL)
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Initialized
         *  - GpuSim_Struct::calcSummationMap() --- Accessed
         *  - GpuSim_Struct::eraseSynapse() --- Modified (= NULL)
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        BGFLOAT ***summationPoint;

        /*! The location of the synapse.
         *  
         *  Usage: NOT USED ANYWHERE
         *  - LIFModel::loadMemory() --- Iniialized
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        Coordinate **synapseCoord;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized (= 0)
         *  - LIFSingleThreadedModel::advanceSynapse() --- Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         *  - GpuSim_Struct::calcSummationMap() --- Accessed
         */
        BGFLOAT **psr;
        
        /*! The decay for the psr.
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::updateDecay() --- Modified
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Modified
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         */
        BGFLOAT **decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::initSpikeQueue() --- Accessed
         *  - LIFSingleThreadedModel::preSpikeHit() --- Accessed
    	 *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
    	 *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         */
        int **total_delay;
        
#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )
        /*! Pointer to the delayed queue
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::initSpikeQueue() --- Initialized
         *  - LIFSingleThreadedModel::preSpikeHit() --- Accessed
         *  - LIFSingleThreadedModel::isSpikeQueue() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized  
    	 *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
    	 *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed
         */
        uint32_t ***delayQueue;
        
        /*! The index indicating the current time slot in the delayed queue
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::initSpikeQueue() --- Initialized
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::preSpikeHit() --- Accessed
         *  - LIFSingleThreadedModel::isSpikeQueue() --- Accessed & Modified
    	 *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
    	 *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified
    	 *  - GpuSim_Struct::createSynapse() --- Initialized
         *  
         *  Note: This variable is used in GpuSim_struct.cu but I am not sure 
         *  if it is actually from a synapse. Will need a little help here. -Aaron
         *  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
         */
        int **delayIdx;
        
        /*! Length of the delayed queue
         *  
         *  Usage: GLOBAL CONSTANT
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::initSpikeQueue() --- Initialized
         *  - LIFSingleThreadedModel::preSpikeHit() --- Accessed
         *  - LIFSingleThreadedModel::isSpikeQueue() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialzied
         */
        int **ldelayQueue;
        
    	/*! Synapse type
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        synapseType **type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::updateDecay() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        BGFLOAT **tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
    	 *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT **r;
        
        /*! The time varying state variable \f$u\f$ for facilitation.
         *  
         *  Usage: LOCAL VARIABL
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
    	 *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT **u;
        
        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
    	 *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT **D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
    	 *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT **U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::createSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
    	 *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT **F;
        
        /*! The time of the last spike.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - LIFSingleThreadedModel::advanceSynapse() --- Accessed & Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
     	 *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified  
         */
        uint64_t **lastSpike;

    	/*! The boolean value indicating the entry in the array is in use.
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeurons() --- Accessed
         *  - LIFSingleThreadedModel::updateWeights() --- Accessed
         *  - LIFSingleThreadedModel::eraseSynapse() --- Modified
    	 *  - LIFSingleThreadedModel::addSynapse() --- Accessed
    	 *  - LIFSingleThreadedModel::createSynapse() --- Modified
         *  - LIFGPUModel::copyDeviceSynapseSumCoordToHost() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Modified
         */
        bool **in_use;

        /*! The number of synapses for each neuron.
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFModel::loadMemory() --- Modified
         *  - LIFModel::saveMemory() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeurons() --- Accessed
    	 *  - LIFSingleThreadedModel::advanceSynapses() --- Accessed
         *  - LIFSingleThreadedModel::updateWeights() --- Accessed
    	 *  - LIFSingleThreadedModel::eraseSynapse() --- Modified
    	 *  - LIFSingleThreadedModel::addSynapse() --- Modified
         *  - LIFGPUModel::copyDeviceSynapseCountsToHost() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Accessed
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
         *  - LIFGPUModel::advance() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Modified
         */
        size_t total_synapse_counts;

    	/*! The maximum number of synapses for each neurons.
         *  
         *  Usage: GLOBAL CONSTANT
         *  - AllSynapses::AllSynapses() --- Initialized
         *  - LIFSingleThreadedModel::addSynapse() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::addSynapse --- Accessed
         *  - GpuSim_Struct::createSynapse --- Accessed
         */
        size_t max_synapses;

    private:
        /*! The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int count_neurons;
};

#endif
