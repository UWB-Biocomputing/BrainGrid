/*
* @file Simulator.cpp
*
* @author Derek McLean
*
* @brief Base class for model-independent simulators targeting different
* platforms.
*/

#include "Simulator.h"

/*
 *  Constructor
 *
 *  @param  model
 *          pointer to a neural network implementation to be simulated by BrainGrid. (It would be
 *          nice if this was a parameter to #simulate). Note: this reference will not be deleted.
 *  @simRecorder        Pointer to the simulation recordig object.
 *  @sInput             Pointer to the stimulus input object.
 *  @param  sim_info    parameters for the simulation.
 */
Simulator::Simulator(IModel *model, IRecorder *simRecorder, ISInput *sInput, SimulationInfo *sim_info) : 
    m_sim_info(sim_info),
    m_model(model), 
    m_simRecorder(simRecorder),
    m_sInput(sInput)
{
    cout << "Neuron count: " << sim_info->totalNeurons << endl;
    g_simulationStep = 0;
}

/*
 * Destructor.
 */
Simulator::~Simulator()
{
    freeResources();
}

/*
 *  Initialize and prepare network for simulation.
 */
void Simulator::setup()
{
    cout << "Initializing models in network." << endl;
    m_model->setupSim(m_sim_info, m_simRecorder);

    // init stimulus input object
    if (m_sInput != NULL)
        m_sInput->init(m_model, *(m_model->getNeurons()), m_sim_info);
}

/*
 *  Begin terminating the simulator.
 */
void Simulator::finish()
{
    // Terminate the simulator
    m_model->cleanupSim(m_sim_info); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/*
 * Resets all of the maps.
 * Releases and re-allocates memory for each map, clearing them as necessary.
 */
void Simulator::reset()
{
    DEBUG(cout << "\nEntering Simulator::reset()" << endl;)

    // Terminate the simulator
    m_model->cleanupSim(m_sim_info);

    // Clean up objects
    freeResources();

    // Reset global simulation Step to 0
    g_simulationStep = 0;

    // Initialize and prepare network for simulation
    m_model->setupSim(m_sim_info, m_simRecorder);

    DEBUG(cout << "\nExiting Simulator::reset()" << endl;)
}

/*
 *  Clean up objects.
 */
void Simulator::freeResources()
{
}

/*
 * Run simulation
 */
void Simulator::simulate()
{
    // Main simulation loop - execute maxGrowthSteps
    for (int currentStep = 1; currentStep <= m_sim_info->maxSteps; currentStep++) {

        DEBUG(cout << endl << endl;)
        DEBUG(cout << "Performing simulation number " << currentStep << endl;)
        DEBUG(cout << "Begin network state:" << endl;)

        // Init SimulationInfo parameters
        m_sim_info->currentStep = currentStep;

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
        m_model->updateConnections(m_sim_info);

        m_model->updateHistory(m_sim_info, m_simRecorder);

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
}

/*
 * Helper for #simulate().
 * Advance simulation until it's ready for the next growth cycle. This should simulate all neuron and
 * synapse activity for one epoch.
 *
 * @param currentStep the current epoch in which the network is being simulated.
 */
void Simulator::advanceUntilGrowth(const int currentStep)
{
    uint64_t count = 0;
    // Compute step number at end of this simulation epoch
    uint64_t endStep = g_simulationStep
            + static_cast<uint64_t>(m_sim_info->epochDuration / m_sim_info->deltaT);

    DEBUG_MID(m_model->logSimStep(m_sim_info);) // Generic model debug call

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

        // input stimulus
        if (m_sInput != NULL)
            m_sInput->inputStimulus(m_model, m_sim_info, m_sim_info->pSummationMap);

	// Advance the Network one time step
        m_model->advance(m_sim_info);

        g_simulationStep++;
    }
}

/*
 * Writes simulation results to an output destination.
 */
void Simulator::saveData() const
{
    m_model->saveData(m_simRecorder);
}

/*
 * Deserializes internal state from a prior run of the simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 * @param memory_in - where to read the state from.
 */
void Simulator::deserialize(istream &memory_in)
{
    // read the neuron data
    memory_in >> m_sim_info->totalNeurons; memory_in.ignore();
    m_model->deserialize(memory_in, m_sim_info);

    // Init history matrices with current values
    m_simRecorder->initValues();
}

/*
 * Serializes internal state for the current simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 * @param memory_out - where to write the state to.
 * This method needs to be debugged to verify that it works.
 */
void Simulator::serialize(ostream &memory_out) const
{
    cerr << "Simulator::writeSimMemory was called. " << endl;
    // get history matrices with current values
    m_simRecorder->getValues();

    m_model->serialize(memory_out, m_sim_info);
}
