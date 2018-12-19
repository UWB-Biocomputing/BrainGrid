/**
 *      @file IAllSynapsesProperties.h
 *
 *      @brief An interface for synapse properties class.
 */

#pragma once

#include "SimulationInfo.h"
#include "ClusterInfo.h"

class IAllSynapsesProperties
{
    public:
        virtual ~IAllSynapsesProperties() {};

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapsesProperties(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;
};
