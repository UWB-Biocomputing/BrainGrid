#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() : size(0)
{
    hasFired = NULL;
    neuron_type_map = NULL;
    spikeCount = NULL;
    spike_history = NULL;
    starter_map = NULL;
    summation_map = NULL;
}

AllNeurons::~AllNeurons()
{
    cleanupNeurons();
}

void AllNeurons::setupNeurons(SimulationInfo *sim_info)
{
    size = sim_info->totalNeurons;
    // TODO: Rename variables for easier identification
    hasFired = new bool[size];
    neuron_type_map = new neuronType[size];
    spikeCount = new int[size];
    starter_map = new bool[size];
    summation_map = new BGFLOAT[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
        spike_history[i] = NULL;
        hasFired[i] = false;
        spikeCount[i] = 0;
    }

    sim_info->pSummationMap = summation_map;
}

void AllNeurons::cleanupNeurons()
{
    if (size != 0) {
        for(int i = 0; i < size; i++) {
            delete[] spike_history[i];
        }
    
        delete[] hasFired;
        delete[] neuron_type_map;
        delete[] spikeCount;
        delete[] starter_map;
        delete[] summation_map;
        delete[] spike_history;
    }
        
    hasFired = NULL;
    neuron_type_map = NULL;
    spikeCount = NULL;
    starter_map = NULL;
    summation_map = NULL;
    spike_history = NULL;

    size = 0;
}

/**
 *  Clear the spike counts out of all Neurons.
 */
//! Clear spike count of each neuron.
void AllNeurons::clearSpikeCounts(const SimulationInfo *sim_info)
{
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        spikeCount[i] = 0;
    }
}
