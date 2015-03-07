#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() : size(0)
{
    neuron_type_map = NULL;
    starter_map = NULL;
    summation_map = NULL;
}

AllNeurons::~AllNeurons()
{
    freeResources();
}

void AllNeurons::setupNeurons(SimulationInfo *sim_info)
{
    size = sim_info->totalNeurons;
    // TODO: Rename variables for easier identification
    neuron_type_map = new neuronType[size];
    starter_map = new bool[size];
    summation_map = new BGFLOAT[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
    }

    sim_info->pSummationMap = summation_map;
}

void AllNeurons::cleanupNeurons()
{
    freeResources();
}

void AllNeurons::freeResources()
{
    if (size != 0) {
        delete[] neuron_type_map;
        delete[] starter_map;
        delete[] summation_map;
    }
        
    neuron_type_map = NULL;
    starter_map = NULL;
    summation_map = NULL;

    size = 0;
}
