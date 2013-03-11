/**
 * @file Simulator.cpp
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 *          platforms.
 */

#include "HostSimulator.h"

HostSimulator::HostSimulator(Network *network, SimulationInfo sim_info) :
    network(network),
    m_sim_info(sim_info)
{

}

HostSimulator::~HostSimulator()
{
    network = NULL;
}

/**
* Run simulation
*
* @param growthStepDuration
* @param maxGrowthSteps
*/
void HostSimulator::simulate()
{
    // Prepare network for simulation.
    // TODO(derek): choose better name after refactor.
    network->setup(m_sim_info.stepDuration, m_sim_info.maxSteps);
    
    // Main simulation loop - execute maxGrowthSteps
    for (int currentStep = 1; currentStep <= m_sim_info.maxSteps; currentStep++) {

        DEBUG(cout << endl << endl;)
        DEBUG(cout << "Performing simulation number " << currentStep << endl;)
        DEBUG(cout << "Begin network state:" << endl;)

        // Advance simulation to next growth cycle
        advanceUntilGrowth(currentStep);

        DEBUG(cout << endl << endl;)
        DEBUG(
            cout << "Done with simulation cycle, beginning growth update "
                 << currentStep << endl;
         )

        // Update the neuron network
#ifdef PERFORMANCE_METRICS
        short_timer.start();
#endif
        network->updateConnections(currentStep);

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
    network->finish(m_sim_info.stepDuration, m_sim_info.maxSteps);
}

void HostSimulator::advanceUntilGrowth(const int currentStep)
{
    // uint64_t count = 0; // TODO what is this?
    uint64_t endStep = g_simulationStep
            + static_cast<uint64_t>(m_sim_info.stepDuration / m_sim_info.deltaT);

    DEBUG2(network->logSimStep()) // Generic model debug call

    while (g_simulationStep < endStep) {
        DEBUG(
            if (count % 10000 == 0) {
                cout << currentStep << "/" << m_sim_info.maxSteps
                     << " simulating time: "
                     << g_simulationStep * m_sim_info.deltaT << endl;
                count = 0;
            }
            count++;
        )

        network->advance();
        g_simulationStep++;
    }
}

void HostSimulator::saveState(ostream &state_out) const
{
    // Write XML header information:
    state_out << "<?xml version=\"1.0\" standalone=\"no\"?>" << endl
       << "<!-- State output file for the DCT growth modeling-->" << endl;
    //state_out << version; TODO: version

    // Write the core state information:
    state_out << "<SimState>" << endl;

    network->saveState(state_out);

    // write time between growth cycles
    state_out << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    state_out << "   " << m_sim_info.stepDuration << endl;
    state_out << "</Matrix>" << endl;

    // write simulation end time
    state_out << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    state_out << "   " << g_simulationStep * m_sim_info.deltaT << endl;
    state_out << "</Matrix>" << endl;
    state_out << "</SimState>" << endl;
}

void HostSimulator::readMemory(istream &memory_in)
{
    network->readSimMemory(memory_in);
}

void HostSimulator::saveMemory(ostream &memory_out) const
{
    network->writeSimMemory(m_sim_info.maxSteps, memory_out);
}

