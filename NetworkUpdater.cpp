/**
 * @file NetworkUpdater.cpp
 *
 * @authors Derek McLean
 *
 * @brief TODO
 */

#include "NetworkUpdater.h"

void NetworkUpdater::NetworkUpdater(int neuron_count)
{
    spikeCounts = new int[neuron_count];
}

/**
 * Compute dist2, dist and delta.
 * @param[in] psi       Pointer to the simulation information.  
 * @param[in] xloc      X location of neurons.
 * @param[in] yloc      Y location of neurons.
 */
/*
void NetworkUpdater::init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc)
{
// MODEL DEPENDENT
    // calculate the distance between neurons
    for (int n = 0; n < psi->cNeurons - 1; n++)
    {
        for (int n2 = n + 1; n2 < psi->cNeurons; n2++)
        {
            // distance^2 between two points in point-slope form
            dist2(n, n2) = (xloc[n] - xloc[n2]) * (xloc[n] - xloc[n2]) +
                (yloc[n] - yloc[n2]) * (yloc[n] - yloc[n2]);

            // both points are equidistant from each other
            dist2(n2, n) = dist2(n, n2);
        }
    }

    // take the square root to get actual distance (Pythagoras was right!)
    // (The CompleteMatrix class makes this assignment look so easy...)
    dist = sqrt(dist2);

    // Init connection frontier distance change matrix with the current distances
    delta = dist;
}
*/

void NetworkUpdater::update(int currentStep, Network *network, SimulationInfo *sim_info)
{
    updateHistory(currentStep, network, sim_info);
    updateRadii(network, sim_info);
    updateFrontiers(sim_info);
    updateOverlap(sim_info);
    updateWeights(sim_info);
}

void NetworkUpdater::updateHistory(int currentStep, Network *network, SimulationInfo *sim_info)
{
    // Calculate growth cycle firing rate for previous period
    network->getSpikeCounts(sim_info->cNeurons, spikeCounts);

    // Calculate growth cycle firing rate for previous period
    for (int i = 0; i < sim_info->cNeurons; i++) {
        // Calculate firing rate
        rates[i] = spikeCounts[i] / sim_info->stepDuration;
        // record firing rate to history matrix
        network->ratesHistory(currentStep, i) = rates[i];
    }

    // clear spike count
    network->clearSpikeCounts(sim_info->cNeurons);

    // compute neuron radii change and assign new values
    outgrowth = 1.0 - 2.0 / (1.0 + exp((sim_info->epsilon - rates / sim_info->maxRate) / sim_info->beta));
    deltaR = sim_info->stepDuration * sim_info->rho * outgrowth;
    radii += deltaR;

    // Cap minimum radius size and record radii to history matrix
    for (int i = 0; i < radii.Size(); i++) {
        // TODO: find out why we cap this here.
        if (radii[i] < sim_info->minRadius) {
            radii[i] = sim_info->minRadius;
        }

        // record radius to history matrix
        network->radiiHistory(sim_info->currentStep, i) = radii[i];

        DEBUG2(cout << "radii[" << i << ":" << radii[i] << "]" << endl;);
    }
}

void NetworkUpdater::updateRadii(Network *network, SimulationInfo *sim_info)
{
    // compute neuron radii change and assign new values
    outgrowth = 1.0 - 2.0 / (1.0 + exp((sim_info->epsilon - rates / sim_info->maxRate) / sim_info->beta));
    deltaR = sim_info->stepDuration * sim_info->rho * outgrowth;
    radii += deltaR;

    // Cap minimum radius size and record radii to history matrix
    for (int i = 0; i < radii.Size(); i++) {
        // TODO: find out why we cap this here.
        if (radii[i] < sim_info->minRadius) {
            radii[i] = sim_info->minRadius;
        }

        // record radius to history matrix
        network->radiiHistory(sim_info->currentStep, i) = radii[i];

        DEBUG2(cout << "radii[" << i << ":" << radii[i] << "]" << endl;);
    }
}

void NetworkUpdater::updateFrontiers(SimulationInfo *sim_info)
{
    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < sim_info->cNeurons - 1; unit++) {
        for (int i = unit + 1; i < sim_info->cNeurons; i++) {
            delta(unit, i) = dist(unit, i) - (radii[unit] + radii[i]);
            delta(i, unit) = delta(unit, i);
        }
    }
}

void NetworkUpdater::updateOverlap(SimulationInfo *sim_info)
{
    DEBUG(cout << "computing areas of overlap" << endl;)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < sim_info->cNeurons; i++) {
        for (int j = 0; j < sim_info->cNeurons; j++) {
            area(i, j) = 0.0;

            if (delta(i, j) < 0) {
                FLOAT lenAB = dist(i, j);
                FLOAT r1 = radii[i];
                FLOAT r2 = radii[j];

                if (lenAB + min(r1, r2) <= max(r1, r2)) {
                    area(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
                    logFile << "Completely overlapping (i, j, r1, r2, area): "
                            << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                } else {
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
}

/**
 * Platform Dependent
 */
void NetworkUpdater::updateWeights(SimulationInfo* sim_info)
{

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
    for (int a = 0; a < sim_info->cNeurons; a++) {
        int xa = a % sim_info->width;
        int ya = a / sim_info->width;
        Coordinate aCoord(xa, ya);

        // and each destination neuron 'b'
        for (int b = 0; b < sim_info->cNeurons; b++) {
            int xb = b % sim_info->width;
            int yb = b / sim_info->width;
            Coordinate bCoord(xb, yb);

            // visit each synapse at (xa,ya)
            bool connected = false;

            // for each existing synapse
            for (size_t syn = 0; syn < sim_info->rgSynapseMap[a].size(); syn++) {
                // if there is a synapse between a and b
                if (sim_info->rgSynapseMap[a][syn]->summationCoord == bCoord) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove 
                    // it from the synapse map if it has gone below 
                    // zero.
                    if (W(a, b) < 0) {
                        removed++;
                        sim_info->rgSynapseMap[a].erase(sim_info->rgSynapseMap[a].begin() + syn);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        sim_info->rgSynapseMap[a][syn]->W = W(a, b) * 
                            synSign(synType(sim_info, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;

                        DEBUG2(cout << "weight of rgSynapseMap" << 
                               coordToString(xa, ya)<<"[" <<syn<<"]: " << 
                               sim_info->rgSynapseMap[a][syn].W << endl;);
                    }
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && (W(a, b) > 0)) {
                added++;

                ISynapse* newSynapse = addSynapse(sim_info, xa, ya, xb, yb);
                newSynapse->W = W(a, b) * synSign(synType(sim_info, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;
            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}
