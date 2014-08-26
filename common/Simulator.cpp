/**
* @file Simulator.cpp
*
* @author Derek McLean
*
* @brief Base class for model-independent simulators targeting different
* platforms.
*/

#include "Simulator.h"

/*
* Constructor
*/
Simulator::Simulator(Network *network, SimulationInfo *sim_info) : network(network), m_sim_info(sim_info)
{
}

/**
* Destructor.
* Releases reference to network.
*/
Simulator::~Simulator()
{
    //network is allocated externally from simulator,
    //do not delete.
    network = NULL;
}

/**
* Run simulation
* @param[in] pInput    Pointer to the stimulus input object.
* @param growthEpochDuration
* @param maxGrowthSteps
*/
void Simulator::simulate(ISInput* pInput)
{
    // Main simulation loop - execute maxGrowthSteps
    // Shouldn't currentStep be an unsigned long?
    for (int currentStep = 1; currentStep <= m_sim_info->maxSteps; currentStep++) {

        DEBUG(cout << endl << endl;)
        DEBUG(cout << "Performing simulation number " << currentStep << endl;)
        DEBUG(cout << "Begin network state:" << endl;)

        // Init SimulationInfo parameters
        m_sim_info->currentStep = currentStep;

        // Advance simulation to next growth cycle
        advanceUntilGrowth(currentStep, pInput);

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
    network->finish(m_sim_info->epochDuration, m_sim_info->maxSteps);
}

/**
* Helper for #simulate().
* Advance simulation until it's ready for the next growth cycle. This should simulate all neuron and
* synapse activity for one epoch.
* @param currentStep the current epoch in which the network is being simulated.
* @param[in] pInput    Pointer to the stimulus input object.
*/
void Simulator::advanceUntilGrowth(const int currentStep, ISInput* pInput)
{
    uint64_t count = 0;
    // Compute step number at end of this simulation epoch
    uint64_t endStep = g_simulationStep
            + static_cast<uint64_t>(m_sim_info->epochDuration / m_sim_info->deltaT);

    DEBUG_MID(network->logSimStep();) // Generic model debug call

    while (g_simulationStep < endStep) {
        DEBUG_LOW(
		  // Output status once every 10,000 steps
            if (count % 10000 == 0)
            {
                cout << currentStep << "/" << m_sim_info->maxSteps
                     << " simulating time: "
                     << g_simulationStep * m_sim_info->deltaT << endl;
                count = 0;
            }
            count++;
        )

	  // Advance the Network one time step
        network->advance(pInput);
        g_simulationStep++;
    }
}

/**
* Writes simulation results to an output destination.
*/
void Simulator::saveState() const
{
    network->saveState();
}

/**
* Deserializes internal state from a prior run of the simulation.
* This allows simulations to be continued from a particular point, to be restarted, or to be
* started from a known state.
* @param memory_in - where to read the state from.
*/
void Simulator::readMemory(istream &memory_in)
{
    network->readSimMemory(memory_in);
}

/**
* Serializes internal state for the current simulation.
* This allows simulations to be continued from a particular point, to be restarted, or to be
* started from a known state.
* @param memory_out - where to write the state to.
* This method needs to be debugged to verify that it works.
*/
void Simulator::saveMemory(ostream &memory_out) const
{
    network->writeSimMemory(memory_out);
}
