#include "AllSpikingNeurons.h"
// Default constructor
AllSpikingNeurons::AllSpikingNeurons() : AllNeurons()
{
    hasFired = NULL;
    spikeCount = NULL;
    spike_history = NULL;
}

AllSpikingNeurons::~AllSpikingNeurons()
{
    freeResources();
}

void AllSpikingNeurons::setupNeurons(SimulationInfo *sim_info)
{
    AllNeurons::setupNeurons(sim_info);

    // TODO: Rename variables for easier identification
    hasFired = new bool[size];
    spikeCount = new int[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        spike_history[i] = NULL;
        hasFired[i] = false;
        spikeCount[i] = 0;
    }

    sim_info->pSummationMap = summation_map;
}

void AllSpikingNeurons::cleanupNeurons()
{
    freeResources();
    AllNeurons::cleanupNeurons();
}

void AllSpikingNeurons::freeResources()
{
    if (size != 0) {
        for(int i = 0; i < size; i++) {
            delete[] spike_history[i];
        }
    
        delete[] hasFired;
        delete[] spikeCount;
        delete[] spike_history;
    }
        
    hasFired = NULL;
    spikeCount = NULL;
    spike_history = NULL;
}

/**
 *  Clear the spike counts out of all Neurons.
 */
//! Clear spike count of each neuron.
void AllSpikingNeurons::clearSpikeCounts(const SimulationInfo *sim_info)
{
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        spikeCount[i] = 0;
    }
}
