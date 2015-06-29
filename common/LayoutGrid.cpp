#include "LayoutGrid.h"

LayoutGrid::LayoutGrid()
{
}

LayoutGrid::~LayoutGrid()
{
}

void LayoutGrid::initNeuronsLocs(const SimulationInfo *sim_info)
{
    int num_neurons = sim_info->totalNeurons;

    // Initialize neuron locations
    for (int i = 0; i < num_neurons; i++) {
        (*xloc)[i] = i % sim_info->width;
        (*yloc)[i] = i / sim_info->width;
    }
}
