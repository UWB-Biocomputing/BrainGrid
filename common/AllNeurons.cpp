#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() : 
        size(0), 
        nParams(0)
{
    summation_map = NULL;
}

AllNeurons::~AllNeurons()
{
    freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllNeurons::setupNeurons(SimulationInfo *sim_info)
{
    size = sim_info->totalNeurons;
    // TODO: Rename variables for easier identification
    summation_map = new BGFLOAT[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
    }

    sim_info->pSummationMap = summation_map;
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllNeurons::cleanupNeurons()
{
    freeResources();
}

/*
 *  Deallocate all resources
 */
void AllNeurons::freeResources()
{
    if (size != 0) {
        delete[] summation_map;
    }
        
    summation_map = NULL;

    size = 0;
}
