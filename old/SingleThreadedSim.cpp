/**
 *      @file SingleThreadedSim.cpp
 *      
 *      @authors Allan Ortiz & Cory Mayberry
 *
 *      @brief A class that performs the single threaded simulation on CPU.
 */
#include "SingleThreadedSim.h"

/** 
 * The constructor for SingleThreadedSim.
 * @post All matrixes are allocated. 
 */
SingleThreadedSim::SingleThreadedSim(SimulationInfo* psi) : HostSim(psi)
{ 
	// Create a normalized random number generator
    rgNormrnd.push_back(new Norm(0, 1, psi->seed));
}

/**
* Destructor
*
*/
SingleThreadedSim::~SingleThreadedSim() 
{ 
}

/**
 * Perform updating neurons and synapses for one activity epoch.
 * @param[in] psi	Pointer to the simulation information.		
 */
void SingleThreadedSim::advanceUntilGrowth(SimulationInfo* psi)
{
    uint64_t count = 0;
    uint64_t endStep = g_simulationStep + static_cast<uint64_t>(psi->stepDuration / psi->deltaT);
    
    DEBUG_MID(printNetworkRadii(radii);)

    while (g_simulationStep < endStep)
    {
        DEBUG_LOW(if (count % 10000 == 0)
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
void SingleThreadedSim::advanceNeurons(SimulationInfo* psi)
{
    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = psi->cNeurons - 1; i >= 0; --i)
    {
        // advance neurons
        (*(psi->pNeuronList))[i]->advance(psi->pSummationMap[i]);

        DEBUG_MID(cout << i << " " << (*(psi->pNeuronList))[i]->Vm << endl;)

        // notify outgoing synapses if neuron has fired
        if ((*(psi->pNeuronList))[i]->hasFired)
        {
            DEBUG_MID(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * psi->deltaT << endl;)

            for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z)            
                psi->rgSynapseMap[i][z]->preSpikeHit();            

            (*(psi->pNeuronList))[i]->hasFired = false;
        }
    }

#ifdef DUMP_VOLTAGES
    // ouput a row with every voltage level for each time step
    cout << g_simulationStep * psi->deltaT;

    for (int i = 0; i < psi->cNeurons; i++)    
        cout << "\t i: " << i << " " << (*(psi->pNeuronList))[i].toStringVm();
    
    cout << endl;
#endif /* DUMP_VOLTAGES */
}

/**
 * @param[in] psi	Pointer to the simulation information.
 */
void SingleThreadedSim::advanceSynapses(SimulationInfo* psi)
{
    for (int i = psi->cNeurons - 1; i >= 0; --i)
    {
        for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z)        
            psi->rgSynapseMap[i][z]->advance();        
    }
}

/**
 * Calculate growth cycle firing rate for previous period.
 * Compute neuron radii change, assign new values, and record the radius to histroy matrix.
 * Update distance between frontiers, and compute areas of overlap. 
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
 * zero.
 * @param[in] psi	Pointer to the simulation information.
 * @param[in] radiiHistory	Matrix to save radius history.
 * @param[in] ratesHistory	Matrix to save firing rates history. 
 */
void SingleThreadedSim::updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory)
{
    // Calculate growth cycle firing rate for previous period
    for (int i = 0; i < psi->cNeurons; i++)
    {
        // Calculate firing rate
        rates[i] = (*(psi->pNeuronList))[i]->getSpikeCount() / psi->stepDuration;

        // clear spike count
        (*(psi->pNeuronList))[i]->clearSpikeCount();

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

        DEBUG_MID(cout << "radii[" << i << ":" << radii[i] << "]" << endl;);
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

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < psi->cNeurons; i++)
    {
        for (int j = 0; j < psi->cNeurons; j++)
        {
            area(i, j) = 0.0;

            if (delta(i, j) < 0)
            {
                BGFLOAT lenAB = dist(i, j);
                BGFLOAT r1 = radii[i];
                BGFLOAT r2 = radii[j];

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
                    BGFLOAT lenAB2 = dist2(i, j);
                    BGFLOAT r12 = r1 * r1;
                    BGFLOAT r22 = r2 * r2;

                    BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                    BGFLOAT angCBA = acos(cosCBA);
                    BGFLOAT angCBD = 2.0 * angCBA;

                    BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                    BGFLOAT angCAB = acos(cosCAB);
                    BGFLOAT angCAD = 2.0 * angCAB;

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
                if (psi->rgSynapseMap[a][syn]->summationCoord == bCoord)
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
                        psi->rgSynapseMap[a][syn]->W = W(a, b) * 
                            synSign(synType(psi, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;

                        DEBUG_MID(cout << "weight of rgSynapseMap" << 
                               coordToString(xa, ya)<<"[" <<syn<<"]: " << 
                               psi->rgSynapseMap[a][syn].W << endl;);
                    }
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && (W(a, b) > 0))
            {
                added++;

                ISynapse* newSynapse = addSynapse(psi, xa, ya, xb, yb);
                newSynapse->W = W(a, b) * synSign(synType(psi, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;
            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}
