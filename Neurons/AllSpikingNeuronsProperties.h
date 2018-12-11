/**
 *      @file AllSpikingNeuronsProperties.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once

#include "AllNeuronsProperties.h"

class AllSpikingNeuronsProperties : public AllNeuronsProperties
{
    public:
        AllSpikingNeuronsProperties();
        virtual ~AllSpikingNeuronsProperties();
   
        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info);

    protected:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupNeuronsProperties();

    public:
        /**
         *  The booleans which track whether the neuron has fired.
         */
        bool *hasFired;

        /**
         *  The number of spikes since the last growth cycle.
         */
        int *spikeCount;

        /**
         *  Offset of the spike_history buffer.
         */
        int *spikeCountOffset;

        /**
         *  Step count (history) for each spike fired by each neuron.
         *  The step counts are stored in a buffer for each neuron, and the pointers
         *  to the buffer are stored in a list pointed by spike_history.
         *  Each buffer is a circular, and offset of top location of the buffer i is
         *  specified by spikeCountOffset[i].
         */
        uint64_t **spike_history;
};
