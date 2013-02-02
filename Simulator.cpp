/**
 * @file Simulator.cpp
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 *          platforms.
 */

#include "Simulator.h"

Simulator::Simulator(Network *network, SimulationInfo sim_info) :
    network(network),
    sim_info(sim_info)
{

}

/**
* Run simulation
*
* @param growthStepDuration
* @param maxGrowthSteps
*/
void Simulator::simulate(FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{
    // Prepare network for simulation.
    // TODO(derek): choose better name after refactor.
    network->setup(growthStepDuration, maxGrowthSteps);
    
    // Main simulation loop - execute maxGrowthSteps
    for (int currentStep = 1; currentStep <= maxGrowthSteps; currentStep++)
    {
#ifdef PERFORMANCE_METRICS
        timer.start();
#endif

        // Init SimulationInfo parameters
        sim_info.currentStep = currentStep;

        DEBUG(cout << "\n\nPerforming simulation number " << currentStep << endl;)
        DEBUG(cout << "Begin network state:" << endl;)

        // Advance simulation to next growth cycle
        network->advanceUntilGrowth(&sim_info);

        DEBUG(cout << "\n\nDone with simulation cycle, beginning growth update " << currentStep << endl;)

        // Update the neuron network
#ifdef PERFORMANCE_METRICS
        short_timer.start();
#endif
        network->update(&sim_info);

#ifdef PERFORMANCE_METRICS
        t_host_adjustSynapses = short_timer.lap() / 1000.0f;
        float total_time = timer.lap() / 1000.0f;
        float t_others = total_time - (t_gpu_rndGeneration + t_gpu_advanceNeurons + 
            t_gpu_advanceSynapses + t_gpu_calcSummation + t_host_adjustSynapses);

        cout << endl;
        cout << "total_time: " << total_time << " ms" << endl;
        printPerformanceMetrics(total_time);
        cout << endl;
#endif
    }
    
    // Tell network to clean-up and run any post-simulation logic.
    // TODO(derek): choose better name after refactor.
    network->finish(growthStepDuration, maxGrowthSteps);
}
