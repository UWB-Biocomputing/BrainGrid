#pragma once

#ifndef _ALLSYNAPSESDEVICE_H_
#define _ALLSYNAPSESDEVICE_H_

#include "Global.h"

struct AllSynapsesDevice
{
    public:
        AllSynapsesDevice();
        AllSynapsesDevice(const int num_neurons, const int max_synapses);
        ~AllSynapsesDevice();
 
        /*! The coordinates of the summation point.
         */
        Coordinate *summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         */
        BGFLOAT *W;

        /*! This synapse's summation point's address.
	 */
        BGFLOAT **summationPoint;

        /*! The location of the synapse.
         */
        Coordinate *synapseCoord;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         */
        BGFLOAT *psr;
        
        /*! The decay for the psr.
         */
        BGFLOAT *decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         */
        int *total_delay;
        
        /*! The delayed queue
         */
        uint32_t *delayQueue;
        
	/*! The index indicating the current time slot in the delayed queue
         */
        int *delayIdx;
        
	/*! Length of the delayed queue
         */
        int *ldelayQueue;
        
    	/*! Synapse type
         */
        synapseType *type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         */
        BGFLOAT *tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         */
        BGFLOAT *r;
        
        /*! The time varying state variable \f$u\f$ for facilitation.
         */
        BGFLOAT *u;
        
        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         */
        BGFLOAT *D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         */
        BGFLOAT *U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         */
        BGFLOAT *F;
        
        /*! The time of the last spike.
         */
        uint64_t *lastSpike;

    	/*! TODO: Define
         */
        bool *in_use;

        /*! The number of synapses for each neuron.
         */
        size_t *synapse_counts;

        /*! The total number of active synapses.
         */
        size_t total_synapse_counts;

    	/*! TODO: Define
         */
        size_t max_synapses;
private:
	uint32_t max_total_synapses;
};

#endif
