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
 */
Simulator::Simulator() 
{
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
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::setup(SimulationInfo *sim_info)
{
  DEBUG(cerr << "Initializing models in network... ";)
  sim_info->model->setupSim(sim_info);
  DEBUG(cerr << "\ndone init models." << endl;)

  // init stimulus input object
  if (sim_info->pInput != NULL) {
    cout << "Initializing input." << endl;
    sim_info->pInput->init(sim_info);
  }
}

/*
 *  Begin terminating the simulator.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::finish(SimulationInfo *sim_info)
{
  // Terminate the simulator
  sim_info->model->cleanupSim(sim_info); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/*
 * Resets all of the maps.
 * Releases and re-allocates memory for each map, clearing them as necessary.
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::reset(SimulationInfo *sim_info)
{
  DEBUG(cout << "\nEntering Simulator::reset()" << endl;)

    // Terminate the simulator
    sim_info->model->cleanupSim(sim_info);

  // Clean up objects
  freeResources();

  // Reset global simulation Step to 0
  g_simulationStep = 0;

  // Initialize and prepare network for simulation
  sim_info->model->setupSim(sim_info);

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
 *
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::simulate(SimulationInfo *sim_info)
{

#ifdef PERFORMANCE_METRICS
  // Start overall simulation timer
  cerr << "Starting main timer... ";
  t_host_adjustSynapses = 0.0;
  timer.start();
  cerr << "done." << endl;
#endif

  // Main simulation loop - execute maxGrowthSteps
  for (int currentStep = 1; currentStep <= sim_info->maxSteps; currentStep++) {

    DEBUG(cout << endl << endl;)
      DEBUG(cout << "Performing simulation number " << currentStep << endl;)
      DEBUG(cout << "Begin network state:" << endl;)

      // Init SimulationInfo parameters
      sim_info->currentStep = currentStep;

    // Advance simulation to next growth cycle
    advanceUntilGrowth(currentStep, sim_info);

    DEBUG(cout << endl << endl;)
      DEBUG(
            cout << "Done with simulation cycle, beginning growth update "
	    << currentStep << endl;
	    )

      // Update the neuron network
#ifdef PERFORMANCE_METRICS
      // Start timer for connection update
      short_timer.start();
#endif
    sim_info->model->updateConnections(sim_info);

    sim_info->model->updateHistory(sim_info);

#ifdef PERFORMANCE_METRICS
    // Times converted from microseconds to seconds
    // Time to update synapses
    t_host_adjustSynapses += short_timer.lap() / 1000000.0;
    // Time since start of simulation
    double total_time = timer.lap() / 1000000.0;

    cout << "\ntotal_time: " << total_time << " seconds" << endl;
    printPerformanceMetrics(total_time, currentStep);
    cout << endl;
#endif
  }
}

/*
 * Helper for #simulate().
 * Advance simulation until it's ready for the next growth cycle. This should simulate all neuron and
 * synapse activity for one epoch.
 *
 *  @param currentStep the current epoch in which the network is being simulated.
 *  @param  sim_info    parameters for the simulation.
 */
void Simulator::advanceUntilGrowth(const int currentStep, SimulationInfo *sim_info)
{
  uint64_t count = 0;
  // Compute step number at end of this simulation epoch
  uint64_t endStep = g_simulationStep
    + static_cast<uint64_t>(sim_info->epochDuration / sim_info->deltaT);

  DEBUG_MID(sim_info->model->logSimStep(sim_info);) // Generic model debug call

    while (g_simulationStep < endStep) {
      DEBUG_LOW(
		// Output status once every 10,000 steps
		if (count % 10000 == 0)
		  {
		    cout << currentStep << "/" << sim_info->maxSteps
			 << " simulating time: "
			 << g_simulationStep * sim_info->deltaT << endl;
		    count = 0;
		  }
		count++;
		)

        // input stimulus
        if (sim_info->pInput != NULL)
	  sim_info->pInput->inputStimulus(sim_info);

      // Advance the Network one time step
      sim_info->model->advance(sim_info);

      g_simulationStep++;
    }
}

/*
 * Writes simulation results to an output destination.
 * 
 *  @param  sim_info    parameters for the simulation. 
 */
void Simulator::saveData(SimulationInfo *sim_info) const
{
  sim_info->model->saveData(sim_info);
}

/*
 * Deserializes internal state from a prior run of the simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 *  @param memory_in - where to read the state from.
 *  @param  sim_info    parameters for the simulation. 
 */
void Simulator::deserialize(istream &memory_in, SimulationInfo *sim_info)
{
  // read the neuron data
  memory_in >> sim_info->totalNeurons; memory_in.ignore();
  sim_info->model->deserialize(memory_in, sim_info);

  // Init history matrices with current values
  sim_info->simRecorder->initValues();
}

/*
 * Serializes internal state for the current simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 * This method needs to be debugged to verify that it works.
 *
 *  @param memory_out - where to write the state to.
 *  @param  sim_info    parameters for the simulation. 
 */
void Simulator::serialize(ostream &memory_out, SimulationInfo *sim_info) const
{
  cerr << "Simulator::writeSimMemory was called. " << endl;
  // get history matrices with current values
  sim_info->simRecorder->getValues();

  sim_info->model->serialize(memory_out, sim_info);
}
