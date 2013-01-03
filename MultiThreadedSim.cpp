/**
 *      @file MultiThreadedSim.cpp
 *      
 *      @author Allan Ortiz & Cory Mayberry
 *
 *      @brief A class that performs the multi threaded simulation on CPU.
 */
#ifdef USE_OMP
#include "MultiThreadedSim.h"

/** 
 * @post All matrixes are allocated. 
 */
MultiThreadedSim::MultiThreadedSim(SimulationInfo* psi) : HostSim(psi)
{
}

MultiThreadedSim::~MultiThreadedSim()
{
}

/**
 * @param[in] psi	Pointer to the simulation information. 	
 */
void MultiThreadedSim::advanceUntilGrowth(SimulationInfo* psi)
{
    uint64_t count = 0;
    uint64_t endStep = g_simulationStep + static_cast<uint64_t>(psi->stepDuration / psi->deltaT);

    cout << "OMP advance" << endl;
    cout << "Thread: " << omp_get_thread_num() << " in par: " << omp_in_parallel() << endl;

    while (g_simulationStep < endStep)
    {
        DEBUG(if (count %1000 == 0)
              {
                  cout << psi->currentStep << "/" << psi->maxSteps
                      << " simulating time: " << g_simulationStep * psi->deltaT << endl;
                  count = 0;
              }

              count++;
             )


        advanceNeurons(psi);

        advanceSynapses(psi);
        g_simulationStep++;
    } 
}

/**
 * Notify outgoing synapses if neuron has fired.
 * @param[in] psi	Pointer to the simulation information. 
 */
void MultiThreadedSim::advanceNeurons(SimulationInfo* psi)
{
    int chunk_size = psi->cNeurons / omp_get_max_threads();

    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    //
    // This loop goes away in a parallel version, as each iteration is a separate thread.
    // In each thread, we advance the neuron associated with the thread, advance its outgoing synapses,
    // and updates a writable copy of the summation map.
    // After all the threads have completed updating the summation map, the "read-only" map is updated with
    // the results of the simulation step. This prevents neurons from interfering with each other's input
    // in a multi-threaded scenario.

#pragma omp parallel for schedule(static, chunk_size) 

    for (int i = psi->cNeurons - 1; i >= 0; --i)
    {
        // advance neurons
        (*(psi->pNeuronList))[i].advance(psi->pSummationMap[i]);

        DEBUG2(cout << i << " " << (*(psi->pNeuronList))[i].Vm << endl;)
    }

    // For the performance reason, this loop should be executed sequentially to avoid critical region for DelayList
    for (int i = psi->cNeurons - 1; i >= 0; --i)
    {
        // notify outgoing synapses if neuron has fired
        if ((*(psi->pNeuronList))[i].hasFired)
        {
            DEBUG2(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * psi->deltaT << endl;)

            for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z)
            {
                psi->rgSynapseMap[i][z].preSpikeHit();
            }

            (*(psi->pNeuronList))[i].hasFired = false;
        }

        // In the parallel version, we would move the inner loop of advanceSynapses 
        // here.
    }

#ifdef DUMP_VOLTAGES
    // ouput a row with every voltage level for each time step
    cout << g_simulationStep * psi->deltaT;

    for (int i = 0; i < psi->cNeurons; i++)
    {
        cout << "\t i: " << i << " " << (*(psi->pNeuronList))[i].toStringVm();
    }

    cout << endl;
#endif /* DUMP_VOLTAGES */
}

/**
 * @param[in] psi	Pointer to the simulation information.
 */
void MultiThreadedSim::advanceSynapses(SimulationInfo* psi)
{
    int chunk_size = psi->cNeurons / omp_get_max_threads();

    // TODO: move global g_nMaxChunkSize into the GPU simulation object
    if (chunk_size < g_nMaxChunkSize)
    {
        chunk_size = psi->cNeurons;
    }

#pragma omp parallel for schedule(static, chunk_size)

    // This code must be inline for OpenMP to work properly.
    for (int i = psi->cNeurons - 1; i >= 0; --i)
    {
        // Advance each synapse to which neuron i outputs
        // GPU/PARALLEL optimization point.
        for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z)
        {
            psi->rgSynapseMap[i][z].advance();
        }
    }
}

/**
 * Calculate growth cycle firing rate for previous period.
 * Compute neuron radii change, assign new values, and record the radius to histroy matrix.
 * Update distance between frontiers, and compute areas of overlap. 
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
 * zero.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] radiiHistory     Matrix to save radius history.
 * @param[in] ratesHistory     Matrix to save firing rates history. 
 */
void MultiThreadedSim::updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory)
{
    //// Calculate OpenMP chunk size
    int max_threads = 1;
    max_threads = omp_get_max_threads();
    int chunk_size = psi->cNeurons / max_threads;
    if (chunk_size < g_nMaxChunkSize)
        chunk_size = psi->cNeurons;

    // Calculate growth cycle firing rate for previous period
    for (int i = 0; i < psi->cNeurons; i++)
    {
        // Calculate firing rate
        rates[i] = (*(psi->pNeuronList))[i].getSpikeCount() / psi->stepDuration;

        // clear spike count
        (*(psi->pNeuronList))[i].clearSpikeCount();

        // record firing rate to history matrix
        ratesHistory(psi->currentStep, i) = rates[i];
    }

    // compute neuron radii change and assign new values
    outgrowth = 1.0 - 2.0 / (1.0 + exp((psi->epsilon - rates / psi->maxRate) / psi->beta));
    deltaR = psi->stepDuration * psi->rho * outgrowth;
    radii += deltaR;

    // Cap minimum radius size and record radii to history matrix
    for (int i = 0; i < radii.Size(); i++)
    {
        // TODO: find out why we cap this here.
        if (radii[i] < psi->minRadius)
            radii[i] = psi->minRadius;

        // record radius to history matrix
        radiiHistory(psi->currentStep, i) = radii[i];

        DEBUG2(cout << "radii[" << i << ":" << radii[i] << "]" << endl;);
    }

    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < psi->cNeurons - 1; unit++)
    {
        for (int i = unit + 1; i < psi->cNeurons; i++)
        {
            delta(unit, i) = dist(unit, i) - (radii[unit] + radii[i]);
            delta(i, unit) = delta(unit, i);
        }
    }

    DEBUG(cout << "computing areas of overlap" << endl;)
#pragma omp parallel for schedule(static, chunk_size)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < psi->cNeurons; i++)
    {
        for (int j = 0; j < psi->cNeurons; j++)
        {
            area(i, j) = 0.0;

            if (delta(i, j) < 0)
            {
                FLOAT lenAB = dist(i, j);
                FLOAT r1 = radii[i];
                FLOAT r2 = radii[j];

                if (lenAB + min(r1, r2) <= max(r1, r2))
                {
                    area(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
                    logFile << "Completely overlapping (i, j, r1, r2, area): "
                    << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                }
                else
                {
                    // Partially overlapping unit
                    FLOAT lenAB2 = dist2(i, j);
                    FLOAT r12 = r1 * r1;
                    FLOAT r22 = r2 * r2;

                    FLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                    FLOAT angCBA = acos(cosCBA);
                    FLOAT angCBD = 2.0 * angCBA;

                    FLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                    FLOAT angCAB = acos(cosCAB);
                    FLOAT angCAD = 2.0 * angCAB;

                    area(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                }
            }
        }
    }

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    W = area;

    int adjusted = 0;
    int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    DEBUG(cout << "adjusting weights" << endl;)

#pragma omp parallel for schedule(static, chunk_size)

    // Scale and add sign to the areas
    // visit each neuron 'a'
    for (int a = 0; a < psi->cNeurons; a++)
    {
        int xa = a % psi->width;
        int ya = a / psi->width;
        Coordinate aCoord(xa, ya);

        // and each destination neuron 'b'
        for (int b = 0; b < psi->cNeurons; b++)
        {
            int xb = b % psi->width;
            int yb = b / psi->width;
            Coordinate bCoord(xb, yb);

            // visit each synapse at (xa,ya)
            bool connected = false;

            // for each existing synapse
            for (size_t syn = 0; syn < psi->rgSynapseMap[a].size(); syn++)
            {
                // if there is a synapse between a and b
                if (psi->rgSynapseMap[a][syn].summationCoord == bCoord)
                {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove 
                    // it from the synapse map if it has gone below 
                    // zero.
                    if (W(a, b) < 0)
                    {
                        removed++;
                        psi->rgSynapseMap[a].erase(psi->rgSynapseMap[a].begin() + syn);
                    }
                    else
                    {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        psi->rgSynapseMap[a][syn].W = W(a, b) * 
                            synSign(synType(psi, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;

                        DEBUG2(cout << "weight of rgSynapseMap" << 
                               coordToString(xa, ya)<<"[" <<syn<<"]: " << 
                               psi->rgSynapseMap[a][syn].W << endl;);
                    }
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && (W(a, b) > 0))
            {
                added++;

                DynamicSpikingSynapse& newSynapse = addSynapse(psi, xa, ya, xb, yb);
                newSynapse.W = W(a, b) * synSign(synType(psi, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;
            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}
#endif //  USE_OMP
