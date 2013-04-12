/**
 * @file Simulator.cpp
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 *          platforms.
 */

#include "HostSimulator.h"

/**
 * Constructor
 *
 * @param network
 *          - pointer to a neural network implementation to be simulated by BrainGrid. (It would be
 *          nice if this was a parameter to #simulate). Note: this reference will not be deleted.
 *          Clients of HostSimulator should handle memory management of the network.
 * @param sim_info - parameters for the simulation.
 */
HostSimulator::HostSimulator(Network *network, SimulationInfo sim_info) :
    network(network),
    m_sim_info(sim_info)
{
    // Create a normalized random number generator
    rgNormrnd.push_back(new Norm(0, 1, sim_info.seed));
}

/**
 * Destructor.
 *
 * Releases reference to network.
 */
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
    DEBUG(cout << "Setup simulation." << endl);
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
    network->finish(m_sim_info.stepDuration, m_sim_info.maxSteps);
}

/**
 * Helper for #simulate().
 *
 * Advance simulation until its ready for the next growth cycle. This should simulate all neuron and
 * synapse activity for one epoch.
 *
 * @param currentStep - the current epoch in which the network is being simulated.
 */
void HostSimulator::advanceUntilGrowth(const int currentStep)
{
    DEBUG(uint64_t count = 0;)
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

/**
 * Writes simulation results to an output destination.
 *
 * @param state_out
 *              - where to write the simulation too (if we are using xml... shouldn't this be an XML
 *              object of some sort?).
 */
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

/**
 * Deserializes internal state from a prior run of the simulation.
 *
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 * @param memory_in - where to read the state from.
 */
void HostSimulator::readMemory(istream &memory_in)
{
    network->readSimMemory(memory_in);
}

/**
 * Serializes internal state for the current simulation.
 *
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 * @param memory_out - where to write the state to.
 */
void HostSimulator::saveMemory(ostream &memory_out) const
{
    network->writeSimMemory(m_sim_info.maxSteps, memory_out);
}
