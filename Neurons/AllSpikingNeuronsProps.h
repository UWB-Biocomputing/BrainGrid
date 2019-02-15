/**
 *      @file AllSpikingNeuronsProps.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once

#include "AllNeuronsProps.h"
#include "Cluster.h"

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

        /**
         *  Clear the spike counts out of all Neurons.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         *  @param  clr       Cluster class to read information from.
         */
        void clearSpikeCounts(const SimulationInfo *sim_info, const ClusterInfo *clr_info, Cluster *clr);

#if defined(USE_GPU)
    public:
        /**
         *  Copy spike history data stored in device memory to host.
         *
         *  @param  allNeuronsDeviceProps   Reference to the AllSpikingNeuronsProps class on device memory.
         *  @param  sim_info                SimulationInfo to refer from.
         *  @param  clr_info                ClusterInfo to refer from.
         */
        void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy spike counts data stored in device memory to host.
         *
         *  @param  allNeuronsDeviceProps   Reference to the AllSpikingNeuronsProps class on device memory.
         *  @param  clr_info                ClusterInfo to refer from.
         */
        void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDeviceProps, const ClusterInfo *clr_info );

    protected:
        /**
         *  Allocate GPU memories to store all neurons' states.
         *
         *  @param  allNeuronsProps   Reference to the AllSpikingNeuronsProps struct.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info               ClusterInfo to refer from.
         */
        void allocNeuronsDeviceProps(AllSpikingNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Delete GPU memories.
         *
         *  @param  allNeuronsProps   Reference to the AllSpikingNeuronsProps class.
         *  @param  clr_info               ClusterInfo to refer from.
         */
        void deleteNeuronsDeviceProps(AllSpikingNeuronsProps &allNeuronsProps, ClusterInfo *clr_info);

        /**
         *  Copy all neurons' data from host to device.
         *  (Helper function of copyNeuronHostToDeviceProps)
         *
         *  @param  allNeuronsProps    Reference to the AllSpikingNeuronsProps class.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        void copyHostToDeviceProps( AllSpikingNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy all neurons' data from device to host.
         *  (Helper function of copyNeuronDeviceToHostProps)
         *
         *  @param  allNeuronsProps    Reference to the AllSpikingNeuronsProps class.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        void copyDeviceToHostProps( AllSpikingNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );
#endif // USE_GPU

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
