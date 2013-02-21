/**
 * @file Simulator.cpp
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 *          platforms.
 */

#include "Simulator.h"

Simulator::Simulator(Network *network, SimulationInfo sim_info,
        bool write_mem_image, ostream& memory_out) :
    network(network),
//    updater(sim_info.cNeurons),
    sim_info(sim_info),
    write_mem_image(write_mem_image),
    memory_out(memory_out)
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
    
    NetworkUpdater *updater = network->getUpdater();
    
    // Main simulation loop - execute maxGrowthSteps
    for (int currentStep = 1; currentStep <= maxGrowthSteps; currentStep++) {

        DEBUG(cout << endl << endl;)
        DEBUG(cout << "Performing simulation number " << currentStep << endl;)
        DEBUG(cout << "Begin network state:" << endl;)

        // Advance simulation to next growth cycle
        advanceUntilGrowth(currentStep, maxGrowthSteps);

        DEBUG(cout << endl << endl;)
        DEBUG(
            cout << "Done with simulation cycle, beginning growth update "
                 << currentStep << endl;
         )

        // Update the neuron network
#ifdef PERFORMANCE_METRICS
        short_timer.start();
#endif
        updater->update(currentStep, network, &sim_info);

#ifdef PERFORMANCE_METRICS
        t_host_adjustSynapses = short_timer.lap() / 1000.0f;
        float total_time = timer.lap() / 1000.0f;
        float t_others = total_time
            - (t_gpu_rndGeneration + t_gpu_advanceNeurons
                + t_gpu_advanceSynapses + t_gpu_calcSummation
                + t_host_adjustSynapses);

        cout << endl;
        cout << "total_time: " << total_time << " ms" << endl;
        printPerformanceMetrics(total_time);
        cout << endl;
#endif
    }

    // Tell network to clean-up and run any post-simulation logic.
    // TODO(derek): choose better name after refactor.
    network->finish(growthStepDuration, maxGrowthSteps);
    
    if (write_mem_image) {
        network->writeSimMemory(maxGrowthSteps, memory_out);
    }
    
    delete updater;
    updater = NULL;
}

void Simulator::advanceUntilGrowth(const int currentStep, const int maxGrowthSteps)
{
    uint64_t count = 0;
    uint64_t endStep = g_simulationStep
            + static_cast<uint64_t>(sim_info.stepDuration / sim_info.deltaT);

    DEBUG2(network->printRadii(&sim_info);)

    while (g_simulationStep < endStep) {
        DEBUG(
            if (count % 10000 == 0) {
                cout << currentStep << "/" << maxGrowthSteps
                     << " simulating time: "
                     << g_simulationStep * sim_info.deltaT << endl;
                count = 0;
            }
            count++;
        )

        network->advanceNeurons(&sim_info);

        network->advanceSynapses(&sim_info);
        g_simulationStep++;
    }
}
