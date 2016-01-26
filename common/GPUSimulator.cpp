#include "GPUSimulator.h"

/*
 *  Constructor
 *  @param  network 
 *          pointer to a neural network implementation to be simulated by BrainGrid. (It would be
 *          nice if this was a parameter to #simulate). Note: this reference will not be deleted.
 *  @param  sim_info    parameters for the simulation.
 */
GPUSimulator::GPUSimulator(Network *network, SimulationInfo *sim_info) : Simulator(network, sim_info)
{
}
