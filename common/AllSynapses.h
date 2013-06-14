#pragma once

#ifndef _ALLSYNAPSES_H_
#define _ALLSYNAPSES_H_

#include "Global.h"

struct AllSynapses
{
    public:
        //! The coordinates of the summation point.
        Coordinate *summationCoord;

        //! The weight (scaling factor, strength, maximal amplitude) of the synapse.
        BGFLOAT *W;

        //! The location of the synapse.
        Coordinate *synapseCoord;

        //! The time step size.
        TIMEFLOAT *deltaT; // must be double for compatibility with GPU code

        //! The post-synaptic response is the result of whatever computation is going on in the synapse.
        BGFLOAT *psr;
        //! The decay for the psr.
        BGFLOAT *decay;
        //! The synaptic transmission delay, descretized into time steps.
        uint32_t *total_delay;
        //! The delayed queue
        uint32_t *delayQueue;
        //! The index indicating the current time slot in the delayed queue
        uint32_t *delay;
        //! Synapse type
        synapseType *type;

        //! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
        BGFLOAT *tau;

        // dynamic synapse vars...........
        //! The time varying state variable \f$r\f$ for depression.
        BGFLOAT *r;
        //! The time varying state variable \f$u\f$ for facilitation.
        BGFLOAT *u;
        //! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
        BGFLOAT *D;
        //! The use parameter of the dynamic synapse [range=(1e-5,1)].
        BGFLOAT *U;
        //! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
        BGFLOAT *F;
        //! The time of the last spike.
        uint64_t *lastSpike;

        GPU_COMPAT_BOOL *in_use; // indicates whethere there is a synapse living here or not (for array(neuron, synapse))

        // The number of synapses for each neuron.
        uint32_t *synapse_counts;

	// The number of neurons
        uint32_t count_neurons;
	
        uint32_t max_synapses;

        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        ~AllSynapses();
};

#endif
