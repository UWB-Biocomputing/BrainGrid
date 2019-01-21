/**
 *      @file AllSpikingNeuronsProps.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once

#include "AllNeuronsProps.h"

class AllSpikingNeuronsProps : public AllNeuronsProps
{
    public:
        AllSpikingNeuronsProps();
        virtual ~AllSpikingNeuronsProps();
   
        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info);

    private:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupNeuronsProps();

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
