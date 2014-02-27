#include "LIFSingleThreadedModel.h"

/*
*  Constructor
*/
LIFSingleThreadedModel::LIFSingleThreadedModel() : LIFModel()
{

}

/*
* Destructor
*/
LIFSingleThreadedModel::~LIFSingleThreadedModel() 
{
	//Let LIFModel base class handle de-allocation
}

/**
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFSingleThreadedModel::advance(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{
    advanceNeurons(neurons, synapses, sim_info);
    advanceSynapses(sim_info.totalNeurons, synapses);
}

/**
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *  @param  currentStep the current step of the simulation.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFSingleThreadedModel::updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{
    updateHistory(currentStep, sim_info.epochDuration, neurons);
    updateFrontiers(sim_info.totalNeurons);
    updateOverlap(sim_info.totalNeurons);
    updateWeights(sim_info.totalNeurons, neurons, synapses, sim_info);
}

/**
 *  Outputs the spikes of the simulation.
 *  Note: only done if STORE_SPIKEHISTORY is true.
 *  @param  neurons list of all Neurons.
 *  @param  sim_info    SimulationInfo to refer.
 */
void LIFSingleThreadedModel::cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info)
{
#ifdef STORE_SPIKEHISTORY
    // output spikes
    for (int i = 0; i < sim_info.width; i++) {
        for (int j = 0; j < sim_info.height; j++) {
            int neuron_index = i + j * sim_info.width;
            uint64_t *pSpikes = neurons.spike_history[neuron_index];

            DEBUG_MID (cout << endl << coordToString(i, j) << endl;);

            for (int i = 0; i < neurons.totalSpikeCount[neuron_index]; i++) {
                DEBUG_MID (cout << i << " ";);
                int idx1 = pSpikes[i] * sim_info.deltaT;
                m_conns->burstinessHist[idx1] = m_conns->burstinessHist[idx1] + 1.0;
                int idx2 = pSpikes[i] * sim_info.deltaT * 100;
                m_conns->spikesHistory[idx2] = m_conns->spikesHistory[idx2] + 1.0;
            }
        }
    }
#endif // STORE_SPIKEHISTORY
}

/**
 *  Log this simulation step.
 *  @param  neurons list of all Neurons
 *  @param  synapses    list of all Synapses
 *  @param  sim_info    SimulationInfo to reference.
 */
void LIFSingleThreadedModel::logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, const SimulationInfo &sim_info) const
{
    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < sim_info.height; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < sim_info.width; x++) {
            switch (neurons.neuron_type_map[x + y * sim_info.width]) {
                case EXC:
                    if (neurons.starter_map[x + y * sim_info.width])
                        ss << "s";
                    else
                        ss << "e";
                    break;
                case INH:
                    ss << "i";
                    break;
                case NTYPE_UNDEF:
                    assert(false);
                    break;
            }

            ss << " " << m_conns->radii[x + y * sim_info.width];
            ss << " " << m_conns->radii[x + y * sim_info.width];

            if (x + 1 < sim_info.width) {
                ss.width(2);
                ss << "|";
                ss.width(2);
            }
        }

        ss << endl;

        for (int i = ss.str().length() - 1; i >= 0; i--) {
            ss << "_";
        }

        ss << endl;
        cout << ss.str();
    }
}

/* -----------------
* # Helper Functions
* ------------------
*/



/**
*  Updates the decay if the synapse selected.
*  @param  synapses    synapse list to find the indexed synapse from.
*  @param  neuron_index    index of the neuron that the synapse belongs to.
*  @param  synapse_index   index of the synapse to set.
*/
bool LIFSingleThreadedModel::updateDecay(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    BGFLOAT &tau = synapses.tau[neuron_index][synapse_index];
    BGFLOAT &deltaT = synapses.deltaT[neuron_index][synapse_index];
    BGFLOAT &decay = synapses.decay[neuron_index][synapse_index];

    if (tau > 0) {
        decay = exp( -deltaT / tau );
        return true;
    }
    return false;
}

/**
 *  Updates the Neuron at the indexed location.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the Neuron that the synapse belongs to.
 */
void LIFSingleThreadedModel::updateNeuron(AllNeurons &neurons, int neuron_index)
{
    BGFLOAT &Tau = neurons.Tau[neuron_index];
    BGFLOAT &C1 = neurons.C1[neuron_index];
    BGFLOAT &C2 = neurons.C2[neuron_index];
    BGFLOAT &deltaT = neurons.deltaT[neuron_index];
    BGFLOAT &Rm = neurons.Rm[neuron_index];
    BGFLOAT &I0 = neurons.I0[neuron_index];
    BGFLOAT &Iinject = neurons.Iinject[neuron_index];
    BGFLOAT &Vrest = neurons.Vrest[neuron_index];

    if (Tau > 0) {
        C1 = exp( -deltaT / Tau );
        C2 = Rm * ( 1 - C1 );
    } else {
        C1 = 0.0;
        C2 = Rm;
    }
    /* calculate const IO */
    if (Rm > 0) {
        I0 = Iinject + Vrest / Rm;
    }else {
        assert(false);
    }
}

/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  neurons the Neuron list to search from
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFSingleThreadedModel::advanceNeurons(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{
    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = sim_info.totalNeurons - 1; i >= 0; --i) {
        // advance neurons
        advanceNeuron(neurons, i);

        // notify outgoing synapses if neuron has fired
        if (neurons.hasFired[i]) {
            DEBUG_MID(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * sim_info.deltaT << endl;)

            for (int z = synapses.synapse_counts[i] - 1; z >= 0; --z) {
                preSpikeHit(synapses, i, z);
            }

            neurons.hasFired[i] = false;
        }
    }

#ifdef DUMP_VOLTAGES
    // ouput a row with every voltage level for each time step
    cout << g_simulationStep * psi->deltaT;

    for (int i = 0; i < psi->cNeurons; i++) {
        cout << "\t i: " << i << " " << m_neuronList[i].toStringVm();
    }
    
    cout << endl;
#endif /* DUMP_VOLTAGES */
}

/**
 *  Update the indexed Neuron.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 */
void LIFSingleThreadedModel::advanceNeuron(AllNeurons &neurons, const int index)
{
    BGFLOAT &Vm = neurons.Vm[index];
    BGFLOAT &Vthresh = neurons.Vthresh[index];
    BGFLOAT &summationPoint = neurons.summation_map[index];
    BGFLOAT &I0 = neurons.I0[index];
    BGFLOAT &Inoise = neurons.Inoise[index];
    BGFLOAT &C1 = neurons.C1[index];
    BGFLOAT &C2 = neurons.C2[index];
    int &nStepsInRefr = neurons.nStepsInRefr[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(neurons, index);
    } else {
        summationPoint += I0; // add IO
        // add noise
        BGFLOAT noise = (*rgNormrnd[0])();
        DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
        summationPoint += noise * Inoise; // add noise
        Vm = C1 * Vm + C2 * summationPoint; // decay Vm and add inputs
    }
    // clear synaptic input for next time step
    summationPoint = 0;

    DEBUG_MID(cout << index << " " << Vm << endl;)
	DEBUG_MID(cout << "NEURON[" << index << "] {" << endl
            << "\tVm = " << Vm << endl
            << "\tVthresh = " << Vthresh << endl
            << "\tsummationPoint = " << summationPoint << endl
            << "\tI0 = " << I0 << endl
            << "\tInoise = " << Inoise << endl
            << "\tC1 = " << C1 << endl
            << "\tC2 = " << C2 << endl
            << "}" << endl
    ;)
}

/**
 *  Prepares Synapse for a spike hit.
 *  @param  synapses    the Synapse list to search from.
 *  @param  neuron_index   index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to update.
 */
void LIFSingleThreadedModel::preSpikeHit(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    uint32_t *delay_queue = synapses.delayQueue[neuron_index][synapse_index];
    int &delayIdx = synapses.delayIdx[neuron_index][synapse_index];
    int &ldelayQueue = synapses.ldelayQueue[neuron_index][synapse_index];
    int &total_delay = synapses.total_delay[neuron_index][synapse_index];

    // Add to spike queue

    // calculate index where to insert the spike into delayQueue
    int idx = delayIdx +  total_delay;
    if ( idx >= ldelayQueue ) {
        idx -= ldelayQueue;
    }

    // set a spike
    assert( !(delay_queue[0] & (0x1 << idx)) );
    delay_queue[0] |= (0x1 << idx);

    delay_queue = NULL;
}

/**
 *  Fire the selected Neuron and calculate the result.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 */
void LIFSingleThreadedModel::fire(AllNeurons &neurons, const int index) const
{
    // Note that the neuron has fired!
    neurons.hasFired[index] = true;

#ifdef STORE_SPIKEHISTORY
    // record spike time
    neurons.spike_history[index][neurons.totalSpikeCount[index]] = g_simulationStep;
#endif // STORE_SPIKEHISTORY

    // increment spike count and total spike count
    neurons.spikeCount[index]++;
    neurons.totalSpikeCount[index]++;

    // calculate the number of steps in the absolute refractory period
    neurons.nStepsInRefr[index] = static_cast<int> ( neurons.Trefract[index] / neurons.deltaT[index] + 0.5 );

    // reset to 'Vreset'
    neurons.Vm[index] = neurons.Vreset[index];
}

/**
 *  Advance all the Synapses in the simulation.
 *  @param  num_neurons number of neurons in the simulation to run.
 *  @param  synapses    list of Synapses to update.
 */
void LIFSingleThreadedModel::advanceSynapses(const int num_neurons, AllSynapses &synapses)
{
    for (int i = num_neurons - 1; i >= 0; --i) {
        for (int z = synapses.synapse_counts[i] - 1; z >= 0; --z) {
            // Advance Synapse
            advanceSynapse(synapses, i, z);
        }
    }
}

/**
 *  Advance one specific Synapse.
 *  @param  synapses    list of the Synapses to advance.
 *  @param  neuron_index    index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to connect to.
 */
void LIFSingleThreadedModel::advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    uint64_t &lastSpike = synapses.lastSpike[neuron_index][synapse_index];
    BGFLOAT &deltaT = synapses.deltaT[neuron_index][synapse_index];
    BGFLOAT &r = synapses.r[neuron_index][synapse_index];
    BGFLOAT &u = synapses.u[neuron_index][synapse_index];
    BGFLOAT &D = synapses.D[neuron_index][synapse_index];
    BGFLOAT &F = synapses.F[neuron_index][synapse_index];
    BGFLOAT &U = synapses.U[neuron_index][synapse_index];
    BGFLOAT &W = synapses.W[neuron_index][synapse_index];
    BGFLOAT &decay = synapses.decay[neuron_index][synapse_index];
    BGFLOAT &psr = synapses.psr[neuron_index][synapse_index];
    BGFLOAT &summationPoint = *(synapses.summationPoint[neuron_index][synapse_index]);

    // is an input in the queue?
    if (isSpikeQueue(synapses, neuron_index, synapse_index)) {
        // adjust synapse parameters
        if (lastSpike != ULONG_MAX) {
            BGFLOAT isi = (g_simulationStep - lastSpike) * deltaT ;
            /*
            DEBUG(
                    cout << "Synapse (" << neuron_index << "," << synapse_index << ") =>"
                         << "r := " << r << " " << flush
                         << "u := " << u << " " << flush
                         << "isi := " << isi << " " << flush
                         << "D := " << D << " " << flush
                         << "U := " << U << " " << flush
                         << "F := " << F
                         << endl;
            )
            */
            r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
            u = U + u * ( 1 - U ) * exp( -isi / F );
        }
        psr += ( ( W / decay ) * u * r );// calculate psr
        lastSpike = g_simulationStep; // record the time of the spike
    }

    // decay the post spike response
    psr *= decay;
    // and apply it to the summation point
#ifdef USE_OMP
#pragma omp atomic
#endif
    summationPoint += psr;
#ifdef USE_OMP
    //PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
    //#pragma omp flush (summationPoint)
#endif
}

/**
 *  Checks if there is an input spike in the queue.
 *  @param  synapses    list of the Synapses to advance.
 *  @param  neuron_index    index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool LIFSingleThreadedModel::isSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    uint32_t *delay_queue = synapses.delayQueue[neuron_index][synapse_index];
    int &delayIdx = synapses.delayIdx[neuron_index][synapse_index];
    int &ldelayQueue = synapses.ldelayQueue[neuron_index][synapse_index];

    bool r = delay_queue[0] & (0x1 << delayIdx);
    delay_queue[0] &= ~(0x1 << delayIdx);
    if ( ++delayIdx >= ldelayQueue ) {
        delayIdx = 0;
    }
    delay_queue = NULL;
    return r;
}

/**
 *  Update the Neuron's history.
 *  @param  currentStep current step of the simulation
 *  @param  epochDuration    duration of the 
 *  @param  neurons the list to update.
 */
void LIFSingleThreadedModel::updateHistory(const int currentStep, BGFLOAT epochDuration, AllNeurons &neurons)
{
    // Calculate growth cycle firing rate for previous period
    //getSpikeCounts(neurons, m_conns->spikeCounts);

    // Calculate growth cycle firing rate for previous period
    for (int i = 0; i < sim_info.totalNeurons; i++) {
        // Calculate firing rate
        m_conns->rates[i] = neurons.spikeCount[i] / epochDuration;
        // record firing rate to history matrix
        m_conns->ratesHistory(currentStep, i) = m_conns->rates[i];
    }

    // clear spike count
    clearSpikeCounts(neurons);

    // compute neuron radii change and assign new values
    m_conns->outgrowth = 1.0 - 2.0 / (1.0 + exp((m_growth.epsilon - m_conns->rates / m_growth.maxRate) / m_growth.beta));
    m_conns->deltaR = epochDuration * m_growth.rho * m_conns->outgrowth;
    m_conns->radii += m_conns->deltaR;

    // Cap minimum radius size and record radii to history matrix
    for (int i = 0; i < m_conns->radii.Size(); i++) {
        // TODO: find out why we cap this here.
        if (m_conns->radii[i] < m_growth.minRadius) {
            m_conns->radii[i] = m_growth.minRadius;
        }

        // record radius to history matrix
        m_conns->radiiHistory(currentStep, i) = m_conns->radii[i];

        DEBUG_MID(cout << "radii[" << i << ":" << m_conns->radii[i] << "]" << endl;);
    }
}

/**
 *  Get the spike counts from all Neurons in the simulation into the given pointer.
 *  @param  neurons the Neuron list to search from.
 *  @param  spikeCounts integer array to fill with the spike counts.
 */
void LIFSingleThreadedModel::getSpikeCounts(const AllNeurons &neurons, int *spikeCounts)
{
    for (int i = 0; i < sim_info.totalNeurons; i++) {
        spikeCounts[i] = neurons.spikeCount[i];
    }
}

/**
 *  Clear the spike counts out of all Neurons.
 *  @param  neurons the Neuron list to search from.
 */
//! Clear spike count of each neuron.
void LIFSingleThreadedModel::clearSpikeCounts(AllNeurons &neurons)
{
    for (int i = 0; i < sim_info.totalNeurons; i++) {
        neurons.spikeCount[i] = 0;
    }
}

/**
 *  Update the distance between frontiers of Neurons.
 *  @param  num_neurons in the simulation to update.
 */
void LIFSingleThreadedModel::updateFrontiers(const int num_neurons)
{
    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < num_neurons - 1; unit++) {
        for (int i = unit + 1; i < num_neurons; i++) {
            m_conns->delta(unit, i) = m_conns->dist(unit, i) - (m_conns->radii[unit] + m_conns->radii[i]);
            m_conns->delta(i, unit) = m_conns->delta(unit, i);
        }
    }
}

/**
 *  Update the areas of overlap in between Neurons.
 *  @param  num_neurons number of Neurons to update.
 */
void LIFSingleThreadedModel::updateOverlap(BGFLOAT num_neurons)
{
    DEBUG(cout << "computing areas of overlap" << endl;)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_neurons; j++) {
            m_conns->area(i, j) = 0.0;

            if (m_conns->delta(i, j) < 0) {
                BGFLOAT lenAB = m_conns->dist(i, j);
                BGFLOAT r1 = m_conns->radii[i];
                BGFLOAT r2 = m_conns->radii[j];

                if (lenAB + min(r1, r2) <= max(r1, r2)) {
                    m_conns->area(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
                    logFile << "Completely overlapping (i, j, r1, r2, area): "
                            << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                } else {
                    // Partially overlapping unit
                    BGFLOAT lenAB2 = m_conns->dist2(i, j);
                    BGFLOAT r12 = r1 * r1;
                    BGFLOAT r22 = r2 * r2;

                    BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                    BGFLOAT angCBA = acos(cosCBA);
                    BGFLOAT angCBD = 2.0 * angCBA;

                    BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                    BGFLOAT angCAB = acos(cosCAB);
                    BGFLOAT angCAD = 2.0 * angCAB;

                    m_conns->area(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                }
            }
        }
    }
}

/**
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void LIFSingleThreadedModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    m_conns->W = m_conns->area;

    int adjusted = 0;
    int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    DEBUG(cout << "adjusting weights" << endl;)

    // Scale and add sign to the areas
    // visit each neuron 'a'
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        int xa = src_neuron % sim_info.width;
        int ya = src_neuron / sim_info.width;
        Coordinate src_coord(xa, ya);

        // and each destination neuron 'b'
        for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
            int xb = dest_neuron % sim_info.width;
            int yb = dest_neuron / sim_info.width;
            Coordinate dest_coord(xb, yb);

            // visit each synapse at (xa,ya)
            bool connected = false;
            synapseType type = synType(neurons, src_neuron, dest_neuron);

            // for each existing synapse
            for (size_t synapse_index = 0; synapse_index < synapses.synapse_counts[src_neuron]; synapse_index++) {
                // if there is a synapse between a and b
                if (synapses.summationCoord[src_neuron][synapse_index] == dest_coord) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.
                    if (m_conns->W(src_neuron, dest_neuron) < 0) {
                        removed++;
                        eraseSynapse(synapses, src_neuron, synapse_index);
                        //sim_info->rgSynapseMap[a].erase(sim_info->rgSynapseMap[a].begin() + syn);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        synapses.W[src_neuron][synapse_index] = m_conns->W(src_neuron, dest_neuron) *
                            synSign(type) * SYNAPSE_STRENGTH_ADJUSTMENT;

                        DEBUG_MID(cout << "weight of rgSynapseMap" <<
                               coordToString(xa, ya)<<"[" <<synapse_index<<"]: " <<
                               synapses.W[src_neuron][synapse_index] << endl;);
                    }
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && (m_conns->W(src_neuron, dest_neuron) > 0)) {

                // locate summation point
                BGFLOAT* sum_point = &( neurons.summation_map[dest_neuron] );
                added++;

                addSynapse(synapses, type, src_neuron, dest_neuron, src_coord, dest_coord, sum_point, sim_info.deltaT);

            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}

/**
 *  Remove a synapse from the network.
 *  @param  neurons the Neuron list to search from.
 *  @param  neuron_index   Index of a neuron.
 *  @param  synapse_index      Index of a synapse.
 */
void LIFSingleThreadedModel::eraseSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    synapses.synapse_counts[neuron_index]--;
    synapses.in_use[neuron_index][synapse_index] = false;
    synapses.summationPoint[neuron_index][synapse_index] = NULL;
}

/**
 *  Adds a Synapse to the model, connecting two Neurons.
 *  @param  synapses    the Neuron list to reference.
 *  @param  type    the type of the Synapse to add.
 *  @param  src_neuron  the Neuron that sends to this Synapse.
 *  @param  dest_neuron the Neuron that receives from the Synapse.
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param  deltaT  TODO
 */
void LIFSingleThreadedModel::addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, BGFLOAT deltaT)
{
    if (synapses.synapse_counts[src_neuron] >= synapses.max_synapses) {
        return; // TODO: ERROR!
    }

    // add it to the list
    size_t synapse_index;
    for (synapse_index = 0; synapse_index < synapses.max_synapses; synapse_index++) {
        if (!synapses.in_use[src_neuron][synapse_index]) {
            break;
        }
    }

    synapses.synapse_counts[src_neuron]++;

    // create a synapse
    createSynapse(synapses, src_neuron, synapse_index, source, dest, sum_point, deltaT, type );
    synapses.W[src_neuron][synapse_index] = m_conns->W(src_neuron, dest_neuron) * synSign(type) * SYNAPSE_STRENGTH_ADJUSTMENT;
}

/**
 *  Create a Synapse and connect it to the model.
 *  @param  synapses    the Neuron list to reference.
 *  @param  neuron_index    TODO 
 *  @param  synapse_index   TODO
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param  deltaT  TODO
 *  @param  type    type of the Synapse to create.
 */
void LIFSingleThreadedModel::createSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, BGFLOAT *sum_point, BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;

    synapses.in_use[neuron_index][synapse_index] = true;
    synapses.summationPoint[neuron_index][synapse_index] = sum_point;
    synapses.summationCoord[neuron_index][synapse_index] = dest;
    synapses.synapseCoord[neuron_index][synapse_index] = source;
    synapses.deltaT[neuron_index][synapse_index] = deltaT;
    synapses.W[neuron_index][synapse_index] = 10.0e-9;
    synapses.psr[neuron_index][synapse_index] = 0.0;
    synapses.delayQueue[neuron_index][synapse_index][0] = 0;
    DEBUG(
        cout << "synapse ("
                << neuron_index << flush
                << ","
                << synapse_index << flush
            << ")"
            << "delay queue length := " << synapses.ldelayQueue[neuron_index][synapse_index] << flush
            << " => " << LENGTH_OF_DELAYQUEUE << endl;
    )
    synapses.ldelayQueue[neuron_index][synapse_index] = LENGTH_OF_DELAYQUEUE;
    synapses.r[neuron_index][synapse_index] = 1.0;
    synapses.u[neuron_index][synapse_index] = 0.4;     // DEFAULT_U
    synapses.lastSpike[neuron_index][synapse_index] = ULONG_MAX;
    synapses.type[neuron_index][synapse_index] = type;

    synapses.U[neuron_index][synapse_index] = DEFAULT_U;
    synapses.tau[neuron_index][synapse_index] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            assert( false );
            break;
    }

    synapses.U[neuron_index][synapse_index] = U;
    synapses.D[neuron_index][synapse_index] = D;
    synapses.F[neuron_index][synapse_index] = F;

    synapses.tau[neuron_index][synapse_index] = tau;
    synapses.total_delay[neuron_index][synapse_index] = static_cast<int>( delay / deltaT ) + 1;
    synapses.decay[neuron_index][synapse_index] = exp( -deltaT / tau );
}

/**
 *  TODO
 *  @param  neurons the Neuron list to search from.
 *  @param  src_coord   the coordinate of the source Neuron.
 *  @param  dest_coord  the coordinate of the destination Neuron.
 *  @param  width   TODO
 */
synapseType LIFSingleThreadedModel::synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const int width)
{
    return synType(neurons, src_coord.x + src_coord.y * width, dest_coord.x + dest_coord.y * width);
}

/**
* Returns the type of synapse at the given coordinates
* @param    neurons the list of all Neurons .
* @param    src_neuron  integer that points to a Neuron in the type map as a source.
* @param    dest_neuron integer that points to a Neuron in the type map as a destination.
* @return   type of synapse at the given coordinate or -1 on error
*/
synapseType LIFSingleThreadedModel::synType(AllNeurons &neurons, const int src_neuron, const int dest_neuron)
{
    if ( neurons.neuron_type_map[src_neuron] == INH && neurons.neuron_type_map[dest_neuron] == INH )
        return II;
    else if ( neurons.neuron_type_map[src_neuron] == INH && neurons.neuron_type_map[dest_neuron] == EXC )
        return IE;
    else if ( neurons.neuron_type_map[src_neuron] == EXC && neurons.neuron_type_map[dest_neuron] == INH )
        return EI;
    else if ( neurons.neuron_type_map[src_neuron] == EXC && neurons.neuron_type_map[dest_neuron] == EXC )
        return EE;

    return STYPE_UNDEF;
}

/**
 *  Get the sign of the synapseType.
 *  @param    type    synapseType I to I, I to E, E to I, or E to E
 *  @return   1 or -1, or 0 if error
 */
int LIFSingleThreadedModel::synSign(const synapseType type)
{
    switch ( type ) {
        case II:
        case IE:
            return -1;
        case EI:
        case EE:
            return 1;
        case STYPE_UNDEF:
            // TODO error.
            return 0;
    }

    return 0;
}








