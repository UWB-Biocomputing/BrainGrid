/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 *  In this file you will find usage statistics for every variable inthe BrainGrid 
 *  project as we find them. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function() (either Modified or Accessed).
 *  
 *  For Example
 *
 *  Usage:
 *  Class::function():			Modified OR Accessed
 *  OtherClass::function():     	Accessed   
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
 
        /*! The coordinates of the summation point.
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFSingleThreadedModel::updateWeights()     Accessed
         *  LIFSingleThreadedModel::createSynapse()     Initialized
         *  GpuSim_Struct::createSynapseImap()          Accessed
         *  GpuSim_Struct::updateNetworkDevice()        Accessed
         *  GpuSim_Struct::createSynapse()              Initialized
         */
        Coordinate **summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::updateWeights()     Modified 
         *  LIFSingleThreadedModel::addSynapse()        Initialized
         *  LIFSingleThreadedModel::createSynapse()     Initialized
         *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
         *  GpuSim_Struct::updateNetworkDevice()        Modified
         *  GpuSim_Struct::addSynapse()                 Initialized
         *  GpuSim_Struct::createSynapse()              Initialized
         *
         *  Note: When searching the GpuSim_struct.cu file for usage statistics
         *  I only searched the document for occurences of "W[". Searching
         *  only "w" lead to every occurance of the letter "w" in the entire 
         *  document. It is possible that usages may have been missed. -Aaron
         */
         BGFLOAT **W;

        /*! This synapse's summation point's address.
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFSingleThreadedModel::createSynapse()	    Initialized
	     *  LIFSingleThreadedModel::advanceSynapse()	Accessed
	     *  LIFSingleThreadedModel::eraseSynapse()	    Modified (= NULL)
	     *  GpuSim_Struct::createSynapse()              Initialized
	     */
        BGFLOAT ***summationPoint;

        /*! The location of the synapse.
         *  
         *  Usage: NOT USED ANYWHERE
         *  LIFSingleThreadedModel::createSynapse()	    Initialized
   	     *  GpuSim_Struct::createSynapse()              Initialized
         */
        Coordinate **synapseCoord;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceSynapse()	Modified
         *  LIFSingleThreadedModel::createSynapse()	    Initialized
         *  GpuSim_Struct::createSynapse()              Initialized
         *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
         *  GpuSim_Struct::calcSummationMap()           Accessed
         */
        BGFLOAT **psr;
        
        /*! The decay for the psr.
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  LIFSingleThreadedModel::updateDecay()	    Initialized
    	 *  LIFSingleThreadedModel::advanceSynapse()	Accessed
	     *  LIFSingleThreadedModel::createSynapse()	    Initialized
         *  GpuSim_Struct::createSynapse()              Initialized
   	     *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
         */
        BGFLOAT **decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed
    	 *  LIFSingleThreadedModel::createSynapse()	    Initialized
         *  GpuSim_Struct::createSynapse()              Initialized
    	 *  GpuSim_Struct::advanceNeuronsDevice()       Accessed
         */
        int **total_delay;
        
#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )
        /*! The delayed queue
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed
         *  LIFSingleThreadedModel::isSpikeQueue()	    Accessed
    	 *  LIFSingleThreadedModel::createSynapse()	    Accessed
         *  GpuSim_Struct::createSynapse()              Modified  
    	 *  GpuSim_Struct::advanceNeuronsDevice()       Modified
    	 *  GpuSim_Struct::advanceSynapseDevice()       Modified
         */
        uint32_t ***delayQueue;
        
	/*! The index indicating the current time slot in the delayed queue
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed
         *  LIFSingleThreadedModel::isSpikeQueue()	    Modified
         *  
         *  Note: This variable is used in GpuSim_struct.cu but I am not sure 
         *  if it is actually from a synapse. Will need a little help here. -Aaron
         *  Note: This variable can be GLOBAL VARIABLE, but need to modify the code.
         */
        int **delayIdx;
        
	/*! Length of the delayed queue
         *  
         *  Usage: GLOBAL CONSTANT
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed
         *  LIFSingleThreadedModel::isSpikeQueue()	    Accessed
    	 *  LIFSingleThreadedModel::createSynapse()     Initialized
         *  GpuSim_Struct::createSynapse()              Initialized
         */
        int **ldelayQueue;
        
    	/*! Synapse type
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFSingleThreadedModel::createSynapse() 	Initialized
         *  GpuSim_Struct::createSynapse()              Initialized
    	 *  GpuSim_Struct::advanceSynapseDevice()       Accessed
         */
        synapseType **type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  LIFSingleThreadedModel::updateDecay() 	    Accessed
	     *  LIFSingleThreadedModel::createSynapse()	    Initialized
	     *  GpuSim_Struct::createSynapse()              Initialized
         */
        BGFLOAT **tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
    	 *  LIFSingleThreadedModel::createSynapse()	    Initialized
 	     *  GpuSim_Struct::createSynapse()              Initialized
    	 *  GpuSim_Struct::advanceSynapseDevice()       Modified
    	 *  
    	 *  Note: When searching the GpuSim_struct.cu file for usage statistics
         *  I only searched the document for occurences of "r[". Searching
         *  only "r" lead to every occurance of the letter "r" in the entire 
         *  document. It is possible that usages may have been missed. -Aaron
         */
        BGFLOAT **r;
        
        /*! The time varying state variable \f$u\f$ for facilitation.
         *  
         *  Usage: LOCAL VARIABL
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
	     *  LIFSingleThreadedModel::createSynapse()	    Initialized
	     *  GpuSim_Struct::createSynapse()              Initialized
    	 *  GpuSim_Struct::advanceSynapseDevice()       Modified
	     *
	     *  Note: When searching the GpuSim_struct.cu file for usage statistics
         *  I only searched the document for occurences of "u[". Searching
         *  only "u" lead to every occurance of the letter "u" in the entire 
         *  document. It is possible that usages may have been missed. -Aaron
         */
        BGFLOAT **u;
        
        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
	     *  LIFSingleThreadedModel::createSynapse()	    Initialized
	     *  LIFSingleThreadedModel::advanceSynapse()	Accessed
	     *  GpuSim_Struct::NOT USED
	     *  Note: Likely under a different name in GpuSim_struct, see synapse_D_d. -Aaron
         */
        BGFLOAT **D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  LIFSingleThreadedModel::advanceSynapse()	Accessed
	     *  LIFSingleThreadedModel::createSynapse()	    Initialized
	     *  
	     *  Note: In GpuSim_struct.cu I cannot differentiate between the
	     *  variables "u" and "U". Likely under a different name in GpuSim_struct, see synapse_U_d.-Aaron
         */
        BGFLOAT **U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  LIFSingleThreadedModel::advanceSynapse()	Accessed
	     *  LIFSingleThreadedModel::createSynapse()	    Initialized
	     *  GpuSim_Struct::NOT USED
	     *
	     *  Note: Likely under a different name in GpuSim_struct, see synapse_F_d. -Aaron
         */
        BGFLOAT **F;
        
        /*! The time of the last spike.
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
    	 *  LIFSingleThreadedModel::createSynapse()	    Initialized
 	     *  GpuSim_Struct::createSynapse()              Initialized
     	 *  GpuSim_Struct::advanceSynapseDevice()       Modified  
         */
        uint64_t **lastSpike;

    	/*! TODO: Define
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::eraseSynapse()	    Modified
    	 *  LIFSingleThreadedModel::addSynapse()	    Accessed
    	 *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  GpuSim_Struct::NOT USED  
         */
        bool **in_use;

        /*! The number of synapses for each neuron.
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceNeurons()	Accessed
    	 *  LIFSingleThreadedModel::advanceSynapses()	Accessed
	     *  LIFSingleThreadedModel::updateWeights()	    Accessed
    	 *  LIFSingleThreadedModel::eraseSynapse()	    Modified
    	 *  LIFSingleThreadedModel::addSynapse()	    Modified
 	     *  GpuSim_Struct::NOT USED  
         * 	     
   	     *  Note: Likely under a different name in GpuSim_struct, see synapse_count. -Aaron
         */
        size_t *synapse_counts;



    	/*! TODO: Define
         *  
         *  Usage: GLOBAL CONSTANT
         *  LIFSingleThreadedModel::addSynapse()	Accessed
 	     *  GpuSim_Struct::NOT USED  
         */
        size_t max_synapses;

        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        ~AllSynapses();

    private:
        /*! The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int count_neurons;
};

#endif
