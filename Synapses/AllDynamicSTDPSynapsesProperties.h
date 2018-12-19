/**
 *      @file AllDynamicSTDPSynapsesProperties.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "AllSTDPSynapsesProperties.h"

class AllDynamicSTDPSynapsesProperties : public AllSTDPSynapsesProperties
{
    public:
        AllDynamicSTDPSynapsesProperties();
        virtual ~AllDynamicSTDPSynapsesProperties();

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
         *  The time of the last spike.
         */
        uint64_t *lastSpike;

        /**
         *  The time varying state variable \f$r\f$ for depression.
         */
        BGFLOAT *r;

        /**
         *  The time varying state variable \f$u\f$ for facilitation.
         */
        BGFLOAT *u;

        /**
         *  The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         */
        BGFLOAT *D;

        /**
         *  The use parameter of the dynamic synapse [range=(1e-5,1)].
         */
        BGFLOAT *U;

        /**
         *  The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         */
        BGFLOAT *F;
};
