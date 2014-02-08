/**
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 *  In this fiel you will find usage statistics for every variable inside the
 *  the LIFCingleThreaded.cpp file. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each function (either Modified or Accessed).
 *  
 *  For Example:
 *
 *  Usage:
 *  function():    Modified OR Accessed
 *  function():     Accessed   
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
         *  updateWeights()     Accessed
         *  createSynapse()     Modified
         */
        Coordinate **summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *  
         *  Usage:
         *  updateWeights()     Modified 
         *  addSynapse()        Modified
         *  createSynapse()     Modified
         */
         BGFLOAT **W;

        /*! This synapse's summation point's address.
         *  
         *  Usage:
         *  createSynapse()	Modified
	 *  advanceNeuron()	Modified
	 *  advanceSynapse()	Modified
	 *  eraseSynapse()	= NULL 
         */
        BGFLOAT ***summationPoint;

        /*! The location of the synapse.
         *  
         *  Usage:
         *  createSynapse()	Modified
         */
        Coordinate **synapseCoord;

        /*! The time step size.
         *  
         *  Usage:
         *  createSynapse()	Modified
	 *  cleanupSim()	Accessed (Sym Info?)
	 *  updateDecay()	Modified
	 *  advanceSynapse()	Modified
         */         
        BGFLOAT **deltaT;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *  
         *  Usage:
	 *  advanceSynapse()	Modified
         *  createSynapse()	Modified (= 0.0)
         */
        BGFLOAT **psr;
        
        /*! The decay for the psr.
         *  
         *  Usage:
         *  updateDecay()	Modified
	 *  advanceSynapse()	Accessed
	 *  createSynapse()	Modified
         */
        BGFLOAT **decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         *  
         *  Usage:
         *  preSpikeHit()	Accessed
	 *  createSynapse()	Modified
         */
        int **total_delay;
        
#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )
        /*! The delayed queue
         *  
         *  Usage:
         *  preSpikeHit()	Modified
         *  isSpikeQueue()	Modified
	 *  createSynapse()	Modified  
         */
        uint32_t ***delayQueue;
        
	/*! The index indicating the current time slot in the delayed queue
         *  
         *  Usage:
         *  preSpikeHit()	Accessed ?
         *  isSpikeQueue()	Modified
         */
        int **delayIdx;
        
	/*! Length of the delayed queue
         *  
         *  Usage:
         *  preSpikeHit()	Accessed
         *  isSpikeQueue()	Modified
	 *  createSynapse()	Modified
         */
        int **ldelayQueue;
        
	/*! Synapse type
         *  
         *  Usage:
         *  createSynapse() 	Accessed 
         */
        synapseType **type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *  
         *  Usage:
         *  updateDecay() 	Accessed
	 *  createSynapse()	Modified
         */
        BGFLOAT **tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *  
         *  Usage:
         *  advanceSynapse()	Modified 
	 *  createSynapse()	Modified
         */
        BGFLOAT **r;
        
        /*! The time varying state variable \f$u\f$ for facilitation.
         *  
         *  Usage:
         *  advanceSynapse()	Modified 
	 *  createSynapse()	Modified
         */
        BGFLOAT **u;
        
        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage:
	 *  createSynapse()	Modified
         */
        BGFLOAT **D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *  
         *  Usage:
         *  advanceSynapse()	Modified 
	 *  createSynapse()	Modified
         */
        BGFLOAT **U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage:
         *  advanceSynapse()	Modified 
	 *  createSynapse()	Modified  
         */
        BGFLOAT **F;
        
        /*! The time of the last spike.
         *  
         *  Usage:
         *  advanceSynapse()	Modified 
	 *  createSynapse()	Modified  
         */
        uint64_t **lastSpike;

	/*! TODO: Define
         *  
         *  Usage:
         *  eraseSynapse()	Modified
	 *  addSynapse()	Accessed
	 *  createSynapse()	Modified
         */
        bool **in_use;

        /*! The number of synapses for each neuron.
         *  
         *  Usage:
         *  advanceNeurons()	Accessed
	 *  advanceSynapses()	Accessed
	 *  updateWeights()	Accessed
	 *  eraseSynapse()	Modified
	 *  addSynapse()	Modified
         */
        size_t *synapse_counts;

	/*! The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage:
         *  Never
         */
        int count_neurons;

	/*! TODO: Define
         *  
         *  Usage:
         *  addSynapse()	Accessed
         */
        size_t max_synapses;

        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        ~AllSynapses();
};

#endif
