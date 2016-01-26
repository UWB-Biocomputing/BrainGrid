#include "SingleThreadedSim.h"



/*
 *  Constructor
 *  @param  network 
 *          pointer to a neural network implementation to be simulated by BrainGrid. (It would be
 *          nice if this was a parameter to #simulate). Note: this reference will not be deleted.
 *          Clients of HostSimulator should handle memory management of the network.
 *  @param  sim_info    parameters for the simulation.
 */
SingleThreadedSim::SingleThreadedSim(Network *network, SimulationInfo *sim_info) : Simulator(network, sim_info)
{

	// Create a normalized random number generator
    rgNormrnd.push_back(new Norm(0, 1, sim_info->seed));
}
