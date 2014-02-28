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
         *  Usage:
         *  LIFSingleThreadedModel::updateWeights()     Accessed
         *  LIFSingleThreadedModel::createSynapse()     Modified
         *  GpuSim_Struct::createSynapseImap()          Accessed
         *  GpuSim_Struct::updateNetworkDevice()        Accessed
         *  GpuSim_Struct::createSynapse()              Modified
         */
        Coordinate **summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::updateWeights()     Modified 
         *  LIFSingleThreadedModel::addSynapse()        Modified
         *  LIFSingleThreadedModel::createSynapse()     Modified
         *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
         *  GpuSim_Struct::updateNetworkDevice()        Modified
         *  GpuSim_Struct::addSynapse()                 Modified
         *  GpuSim_Struct::createSynapse()              Modified
         *
         *  Note: When searching the GpuSim_struct.cu file for usage statistics
         *  I only searched the document for occurences of "W[". Searching
         *  only "w" lead to every occurance of the letter "w" in the entire 
         *  document. It is possible that usages may have been missed. -Aaron
         */
         BGFLOAT **W;

        /*! This synapse's summation point's address.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  LIFSingleThreadedModel::advanceNeuron()	    Modified ?
	     *  LIFSingleThreadedModel::advanceSynapse()	Modified ?
	     *  LIFSingleThreadedModel::eraseSynapse()	    Modified (= NULL)
	     *  GpuSim_Struct::createSynapse()              Modified
	     */
        BGFLOAT ***summationPoint;

        /*! The location of the synapse.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::createSynapse()	    Modified
   	     *  GpuSim_Struct::createSynapse()              Modified
         */
        Coordinate **synapseCoord;

        /*! The time step size.
	  *
         *  Why isn't this just the global step size?
         *  Usage:
         *  LIFSingleThreadedModel::createSynapse()	Modified
    	 *  LIFSingleThreadedModel::cleanupSim()	Accessed (Sym Info?)
    	 *  LIFSingleThreadedModel::updateDecay()	Modified
         *  LIFSingleThreadedModel::advanceSynapse()	Modified
         *  GpuSim_Struct::createSynapse()              Modified
         *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
         */         
        BGFLOAT **deltaT;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *  
         *  Usage:
	 *  LIFSingleThreadedModel::advanceSynapse()	Modified
         *  LIFSingleThreadedModel::createSynapse()	Modified
         *  GpuSim_Struct::createSynapse()              Modified
   	 *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
   	 *  GpuSim_Struct::calcSummationMap()           Accessed
         */
        BGFLOAT **psr;
        
        /*! The decay for the psr.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::updateDecay()	    Modified
    	 *  LIFSingleThreadedModel::advanceSynapse()	Accessed
	     *  LIFSingleThreadedModel::createSynapse()	    Modified
         *  GpuSim_Struct::createSynapse()              Modified
   	     *  GpuSim_Struct::advanceSynapsesDevice()      Accessed
         */
        BGFLOAT **decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed
    	 *  LIFSingleThreadedModel::createSynapse()	    Modified
         *  GpuSim_Struct::createSynapse()              Modified
    	 *  GpuSim_Struct::advanceNeuronsDevice()       Accessed
         */
        int **total_delay;
        
#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )
        /*! The delayed queue
         *  
         *  Usage:
         *  LIFSingleThreadedModel::preSpikeHit()	    Modified
         *  LIFSingleThreadedModel::isSpikeQueue()	    Modified
    	 *  LIFSingleThreadedModel::createSynapse()	    Modified
         *  GpuSim_Struct::createSynapse()              Modified  
    	 *  GpuSim_Struct::advanceNeuronsDevice()       Modified
    	 *  GpuSim_Struct::advanceSynapseDevice()       Modified
         */
        uint32_t ***delayQueue;
        
	/*! The index indicating the current time slot in the delayed queue
         *  
         *  Usage:
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed ?
         *  LIFSingleThreadedModel::isSpikeQueue()	    Modified
         *  
         *  Note: This variable is used in GpuSim_struct.cu but I am not sure 
         *  if it is actually from a synapse. Will need a little help here. -Aaron
         */
        int **delayIdx;
        
	/*! Length of the delayed queue
         *  
         *  Usage:
         *  LIFSingleThreadedModel::preSpikeHit()	    Accessed
         *  LIFSingleThreadedModel::isSpikeQueue()	    Modified
    	 *  LIFSingleThreadedModel::createSynapse()     Modified
         *  GpuSim_Struct::createSynapse()              Modified  
         */
        int **ldelayQueue;
        
    	/*! Synapse type
         *  
         *  Usage:
         *  LIFSingleThreadedModel::createSynapse() 	Accessed 
         *  GpuSim_Struct::createSynapse()              Modified  
    	 *  GpuSim_Struct::advanceSynapseDevice()       Accessed
         */
        synapseType **type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *  
         *  Usage:
         *  LIFSingleThreadedModel::updateDecay() 	    Accessed
	     *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  GpuSim_Struct::createSynapse()              Modified  
         */
        BGFLOAT **tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
    	 *  LIFSingleThreadedModel::createSynapse()	    Modified
 	     *  GpuSim_Struct::createSynapse()              Modified  
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
         *  Usage:
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
	     *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  GpuSim_Struct::createSynapse()              Modified  
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
         *  Usage:
	     *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  GpuSim_Struct::NOT USED
	     *  Note: Likely under a different name in GpuSim_struct, see synapse_D_d. -Aaron
         */
        BGFLOAT **D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *  
         *  Usage:
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
	     *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  
	     *  Note: In GpuSim_struct.cu I cannot differentiate between the
	     *  variables "u" and "U". Likely under a different name in GpuSim_struct, see synapse_U_d.-Aaron
         */
        BGFLOAT **U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage:
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
	     *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  GpuSim_Struct::NOT USED
	     *
	     *  Note: Likely under a different name in GpuSim_struct, see synapse_F_d. -Aaron
         */
        BGFLOAT **F;
        
        /*! The time of the last spike.
         *  
         *  Usage:
         *  LIFSingleThreadedModel::advanceSynapse()	Modified 
    	 *  LIFSingleThreadedModel::createSynapse()	    Modified
 	     *  GpuSim_Struct::createSynapse()              Modified  
     	 *  GpuSim_Struct::advanceSynapseDevice()       Modified  
         */
        uint64_t **lastSpike;

    	/*! TODO: Define
         *  
         *  Usage:
         *  LIFSingleThreadedModel::eraseSynapse()	    Modified
    	 *  LIFSingleThreadedModel::addSynapse()	    Accessed
    	 *  LIFSingleThreadedModel::createSynapse()	    Modified
	     *  GpuSim_Struct::NOT USED  
         */
        bool **in_use;

        /*! The number of synapses for each neuron.
         *  
         *  Usage:
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

    	/*! The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int count_neurons;

    	/*! TODO: Define
         *  
         *  Usage:
         *  LIFSingleThreadedModel::addSynapse()	Accessed
 	     *  GpuSim_Struct::NOT USED  
         */
        size_t max_synapses;

        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        ~AllSynapses();
};

#endif
