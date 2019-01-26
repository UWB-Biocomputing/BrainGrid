/**
 *      @file AllSpikingSynapsesProps.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "AllSynapsesProps.h"

class AllSpikingSynapsesProps : public AllSynapsesProps
{
    public:
        AllSpikingSynapsesProps();
        virtual ~AllSpikingSynapsesProps();

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Sets the data for Synapse to input's data.
         *
         *  @param  input  istream to read from.
         *  @param  iSyn   Index of the synapse to set.
         */
        virtual void readSynapseProps(istream &input, const BGSIZE iSyn);

        /**
         *  Write the synapse data to the stream.
         *
         *  @param  output  stream to print out to.
         *  @param  iSyn    Index of the synapse to print out.
         */
        virtual void writeSynapseProps(ostream& output, const BGSIZE iSyn) const;

    private:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupSynapsesProps();

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
