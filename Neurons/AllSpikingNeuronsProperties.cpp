#include "AllSpikingNeuronsProperties.h"

// Default constructor
AllSpikingNeuronsProperties::AllSpikingNeuronsProperties() : AllNeuronsProperties()
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

AllSpikingNeuronsProperties::~AllSpikingNeuronsProperties()
{
    cleanupNeuronsProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingNeuronsProperties::setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllNeuronsProperties::setupNeuronsProperties(sim_info, clr_info);

    // TODO: Rename variables for easier identification
    hasFired = new bool[size];
    spikeCount = new int[size];
    spikeCountOffset = new int[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        spike_history[i] = NULL;
        hasFired[i] = false;
        spikeCount[i] = 0;
        spikeCountOffset[i] = 0;
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingNeuronsProperties::cleanupNeuronsProperties()
{
    if (size != 0) {
        for(int i = 0; i < size; i++) {
            delete[] spike_history[i];
        }

        delete[] hasFired;
        delete[] spikeCount;
        delete[] spikeCountOffset;
        delete[] spike_history;
    }

    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;

    AllNeuronsProperties::cleanupNeuronsProperties();
}
