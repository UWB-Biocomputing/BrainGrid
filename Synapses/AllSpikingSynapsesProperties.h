/**
 *      @file AllSpikingSynapsesProperties.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "AllSynapsesProperties.h"

class AllSpikingSynapsesProperties : public AllSynapsesProperties
{
    public:
        AllSpikingSynapsesProperties();
        virtual ~AllSpikingSynapsesProperties();

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapsesProperties(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info);

    protected:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        virtual void cleanupSynapsesProperties();

    public:
        /**
         *  The decay for the psr.
         */
        BGFLOAT *decay;

        /**
         *  The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         */
        BGFLOAT *tau;

        /**
         *  The synaptic transmission delay, descretized into time steps.
         */
        int *total_delay;

        /**
         * The collection of synaptic transmission delay queue.
         */
        EventQueue *preSpikeQueue;
};
