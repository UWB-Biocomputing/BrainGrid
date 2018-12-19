/**
 *      @file AllSynapsesProperties.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "IAllSynapsesProperties.h"

class AllSynapsesProperties : public IAllSynapsesProperties
{
    public:
        AllSynapsesProperties();
        virtual ~AllSynapsesProperties();

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
         *  The location of the source neuron
         */
        int *sourceNeuronLayoutIndex;

        /**
         *  The location of the destination neuron
         */
        int *destNeuronLayoutIndex;

        /**
         *   The weight (scaling factor, strength, maximal amplitude) of the synapse.
         */
         BGFLOAT *W;

        /**
         *  This synapse's summation point's address.
         */
        BGFLOAT **summationPoint;

        /**
         *  Synapse type
         */
        synapseType *type;

        /**
         *  The post-synaptic response is the result of whatever computation
         *  is going on in the synapse.
         */
        BGFLOAT *psr;

        /**
         *  The boolean value indicating the entry in the array is in use.
         */
        bool *in_use;

        /**
         *  The number of synapses for each neuron.
         *  Note: Likely under a different name in GpuSim_struct, see synapse_count. -Aaron
         */
        BGSIZE *synapse_counts;

        /**
         *  The total number of active synapses.
         */
        BGSIZE total_synapse_counts;

        /**
         *  The maximum number of synapses for each neurons.
         */
        BGSIZE maxSynapsesPerNeuron;

        /**
         *  The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int count_neurons;

        /**
         *  A temporary variable used for parallel reduction in calcSummationMapDevice.
         */
        BGFLOAT *summation;
};
