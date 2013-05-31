#pragma once

#ifndef _ALLSYNAPSES_H_
#define _ALLSYNAPSES_H_

#include "Global.h"

struct AllSynapses
{
    public:
        //! The coordinates of the summation point.
        vector<Coordinate> summationCoord;

        //! The weight (scaling factor, strength, maximal amplitude) of the synapse.
        vector<BGFLOAT> W;

        //! The location of the synapse.
        vector<Coordinate> synapseCoord;

        //! The time step size.
        vector<TIMEFLOAT> deltaT; // must be double for compatibility with GPU code

        //! The post-synaptic response is the result of whatever computation is going on in the synapse.
        vector<BGFLOAT> psr;
        //! The decay for the psr.
        vector<BGFLOAT> decay;
        //! The synaptic transmission delay, descretized into time steps.
        vector<uint32_t> total_delay;
        //! The delayed queue
        vector<uint32_t> delayQueue;
        //! The index indicating the current time slot in the delayed queue
        vector<uint32_t> delayIdx;
        //! Length of the delayed queue
        vector<uint32_t> ldelayQueue;
        //! Synapse type
        vector<synapseType> type;

        //! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
        vector<BGFLOAT> tau;

        // dynamic synapse vars...........
        //! The time varying state variable \f$r\f$ for depression.
        vector<BGFLOAT> r;
        //! The time varying state variable \f$u\f$ for facilitation.
        vector<BGFLOAT> u;
        //! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
        vector<BGFLOAT> D;
        //! The use parameter of the dynamic synapse [range=(1e-5,1)].
        vector<BGFLOAT> U;
        //! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
        vector<BGFLOAT> F;
        //! The time of the last spike.
        vector<uint64_t> lastSpike;

        vector<GPU_COMPAT_BOOL> in_use; // indicates whethere there is a synapse living here or not (for array(neuron, synapse))

        // The number of synapses for each neuron.
        vector<size_t> synapse_counts;

	// The number of neurons
        int count_neurons;

	
        size_t max_synapses;

        AllSynapses();
        AllSynapses(const int num_neurons, const int max_synapses);
        ~AllSynapses();
};

#endif
