#include "LIFSingleThreadedModel.h"
#include "AllLIFNeurons.h"
#include "AllDSSynapses.h"

/*
*  Constructor
*/
LIFSingleThreadedModel::LIFSingleThreadedModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 
    Model(conns, neurons, synapses, layout)
{
}

/*
* Destructor
*/
LIFSingleThreadedModel::~LIFSingleThreadedModel() 
{
	//Let Model base class handle de-allocation
}

/**
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFSingleThreadedModel::advance(const SimulationInfo *sim_info)
{
    advanceNeurons(*m_neurons, *m_synapses, sim_info);
    advanceSynapses(sim_info->totalNeurons, *m_synapses, sim_info->deltaT);
}

/**
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *  @param  currentStep the current step of the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void LIFSingleThreadedModel::updateConnections(const int currentStep, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
    const int num_neurons = sim_info->totalNeurons;
    updateHistory(currentStep, sim_info->epochDuration, sim_info, simRecorder);
    // Update the distance between frontiers of Neurons
    m_conns->updateFrontiers(num_neurons);
    // Update the areas of overlap in between Neurons
    m_conns->updateOverlap(num_neurons);
    updateWeights(num_neurons, *m_neurons, *m_synapses, sim_info);
}

/* -----------------
* # Helper Functions
* ------------------
*/


/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  neurons the Neuron list to search from
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFSingleThreadedModel::advanceNeurons(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{
    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = sim_info->totalNeurons - 1; i >= 0; --i) {
        // advance neurons
        advanceNeuron(neurons, i, sim_info->deltaT);

        // notify outgoing synapses if neuron has fired
        if (neurons.hasFired[i]) {
            DEBUG_MID(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * sim_info->deltaT << endl;)

            int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
            assert( neurons.spikeCount[i] < max_spikes );

            size_t synapse_counts = synapses.synapse_counts[i];
            int synapse_notified = 0;
            for (int z = 0; synapse_notified < synapse_counts; z++) {
                if (synapses.in_use[i][z] == true) {
                    preSpikeHit(synapses, i, z);
                    synapse_notified++;
                }
            }

            neurons.hasFired[i] = false;
        }
    }

#ifdef DUMP_VOLTAGES
    // ouput a row with every voltage level for each time step
    cout << g_simulationStep * sim_info->deltaT;

    for (int i = 0; i < sim_info->totalNeurons; i++) {
        cout << "\t i: " << i << " " << m_neuronList[i].toStringVm();
    }
    
    cout << endl;
#endif /* DUMP_VOLTAGES */
}

/**
 *  Update the indexed Neuron.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 *  @param  deltaT  inner simulation step duration.
 */
void LIFSingleThreadedModel::advanceNeuron(AllNeurons &neurons, const int index, const BGFLOAT deltaT)
{
    AllLIFNeurons &LifNeurons = dynamic_cast<AllLIFNeurons&>(neurons);
    BGFLOAT &Vm = LifNeurons.Vm[index];
    BGFLOAT &Vthresh = LifNeurons.Vthresh[index];
    BGFLOAT &summationPoint = LifNeurons.summation_map[index];
    BGFLOAT &I0 = LifNeurons.I0[index];
    BGFLOAT &Inoise = LifNeurons.Inoise[index];
    BGFLOAT &C1 = LifNeurons.C1[index];
    BGFLOAT &C2 = LifNeurons.C2[index];
    int &nStepsInRefr = LifNeurons.nStepsInRefr[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(neurons, index, deltaT);
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
 *  @param  deltaT  inner simulation step duration
 */
void LIFSingleThreadedModel::fire(AllNeurons &neurons, const int index, const BGFLOAT deltaT) const
{
    // Note that the neuron has fired!
    neurons.hasFired[index] = true;

    // record spike time
    neurons.spike_history[index][neurons.spikeCount[index]] = g_simulationStep;

    // increment spike count and total spike count
    neurons.spikeCount[index]++;

    // calculate the number of steps in the absolute refractory period
    AllLIFNeurons &LifNeurons = dynamic_cast<AllLIFNeurons&>(neurons);
    LifNeurons.nStepsInRefr[index] = static_cast<int> ( LifNeurons.Trefract[index] / deltaT + 0.5 );

    // reset to 'Vreset'
    LifNeurons.Vm[index] = LifNeurons.Vreset[index];
}

/**
 *  Advance all the Synapses in the simulation.
 *  @param  num_neurons number of neurons in the simulation to run.
 *  @param  synapses    list of Synapses to update.
 *  @param  deltaT      inner simulation step duration
 */
void LIFSingleThreadedModel::advanceSynapses(const int num_neurons, AllSynapses &synapses, const BGFLOAT deltaT)
{
    for (int i = 0; i < num_neurons; i++) {
        size_t synapse_counts = synapses.synapse_counts[i];
        int synapse_advanced = 0;
        for (int z = 0; z < synapse_counts; z++) {
            // Advance Synapse
            advanceSynapse(synapses, i, z, deltaT);
            synapse_advanced++;
        }
    }
}

/**
 *  Advance one specific Synapse.
 *  @param  synapses    list of the Synapses to advance.
 *  @param  neuron_index    index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to connect to.
 *  @param  deltaT   inner simulation step duration
 */
void LIFSingleThreadedModel::advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, const BGFLOAT deltaT)
{
    uint64_t &lastSpike = synapses.lastSpike[neuron_index][synapse_index];
    AllDSSynapses &DsSynapses = dynamic_cast<AllDSSynapses&>(synapses);
    BGFLOAT &r = DsSynapses.r[neuron_index][synapse_index];
    BGFLOAT &u = DsSynapses.u[neuron_index][synapse_index];
    BGFLOAT &D = DsSynapses.D[neuron_index][synapse_index];
    BGFLOAT &F = DsSynapses.F[neuron_index][synapse_index];
    BGFLOAT &U = DsSynapses.U[neuron_index][synapse_index];
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
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void LIFSingleThreadedModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    (*m_conns->W) = (*m_conns->area);

    int adjusted = 0;
    int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    DEBUG(cout << "adjusting weights" << endl;)

    // Scale and add sign to the areas
    // visit each neuron 'a'
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        int xa = src_neuron % sim_info->width;
        int ya = src_neuron / sim_info->width;
        Coordinate src_coord(xa, ya);

        // and each destination neuron 'b'
        for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
            int xb = dest_neuron % sim_info->width;
            int yb = dest_neuron / sim_info->width;
            Coordinate dest_coord(xb, yb);

            // visit each synapse at (xa,ya)
            bool connected = false;
            synapseType type = synType(neurons, src_neuron, dest_neuron);

            // for each existing synapse
            size_t synapse_counts = synapses.synapse_counts[src_neuron];
            int synapse_adjusted = 0;
            for (size_t synapse_index = 0; synapse_adjusted < synapse_counts; synapse_index++) {
                if (synapses.in_use[src_neuron][synapse_index] == true) {
                    // if there is a synapse between a and b
                    if (synapses.summationCoord[src_neuron][synapse_index] == dest_coord) {
                        connected = true;
                        adjusted++;

                        // adjust the strength of the synapse or remove
                        // it from the synapse map if it has gone below
                        // zero.
                        if ((*m_conns->W)(src_neuron, dest_neuron) < 0) {
                            removed++;
                            eraseSynapse(synapses, src_neuron, synapse_index);
                        } else {
                            // adjust
                            // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                            synapses.W[src_neuron][synapse_index] = (*m_conns->W)(src_neuron, dest_neuron) *
                                synSign(type) * AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

                            DEBUG_MID(cout << "weight of rgSynapseMap" <<
                                   coordToString(xa, ya)<<"[" <<synapse_index<<"]: " <<
                                   synapses.W[src_neuron][synapse_index] << endl;);
                        }
                    }
                    synapse_adjusted++;
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && ((*m_conns->W)(src_neuron, dest_neuron) > 0)) {

                // locate summation point
                BGFLOAT* sum_point = &( neurons.summation_map[dest_neuron] );
                added++;

                addSynapse(synapses, type, src_neuron, dest_neuron, src_coord, dest_coord, sum_point, sim_info->deltaT);

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
 *  @param deltaT   inner simulation step duration
 */
void LIFSingleThreadedModel::addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, const BGFLOAT deltaT)
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
    synapses.createSynapse(src_neuron, synapse_index, source, dest, sum_point, deltaT, type );
    synapses.W[src_neuron][synapse_index] = (*m_conns->W)(src_neuron, dest_neuron) 
            * synSign(type) * AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
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
