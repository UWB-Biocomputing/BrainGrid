#include "SingleThreadedSpikingModel.h"
#include "AllDSSynapses.h"

/*
*  Constructor
*/
SingleThreadedSpikingModel::SingleThreadedSpikingModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 
    Model(conns, neurons, synapses, layout)
{
}

/*
* Destructor
*/
SingleThreadedSpikingModel::~SingleThreadedSpikingModel() 
{
	//Let Model base class handle de-allocation
}

/**
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::advance(const SimulationInfo *sim_info)
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
void SingleThreadedSpikingModel::updateConnections(const int currentStep, const SimulationInfo *sim_info, IRecorder* simRecorder)
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
void SingleThreadedSpikingModel::advanceNeurons(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{
    AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons&>(neurons);

    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = sim_info->totalNeurons - 1; i >= 0; --i) {
        // advance neurons
        advanceNeuron(neurons, i, sim_info->deltaT);

        // notify outgoing synapses if neuron has fired
        if (spNeurons.hasFired[i]) {
            DEBUG_MID(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * sim_info->deltaT << endl;)

            int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
            assert( spNeurons.spikeCount[i] < max_spikes );

            size_t synapse_counts = synapses.synapse_counts[i];
            int synapse_notified = 0;
            for (int z = 0; synapse_notified < synapse_counts; z++) {
                uint32_t iSyn = sim_info->maxSynapsesPerNeuron * i + z;
                if (synapses.in_use[iSyn] == true) {
                    preSpikeHit(synapses, iSyn);
                    synapse_notified++;
                }
            }

            spNeurons.hasFired[i] = false;
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
 *  Prepares Synapse for a spike hit.
 *  @param  synapses    the Synapse list to search from.
 *  @param  iSyn   index of the Synapse to update.
 */
void SingleThreadedSpikingModel::preSpikeHit(AllSynapses &synapses, const uint32_t iSyn)
{
    uint32_t &delay_queue = synapses.delayQueue[iSyn];
    int &delayIdx = synapses.delayIdx[iSyn];
    int &ldelayQueue = synapses.ldelayQueue[iSyn];
    int &total_delay = synapses.total_delay[iSyn];

    // Add to spike queue

    // calculate index where to insert the spike into delayQueue
    int idx = delayIdx +  total_delay;
    if ( idx >= ldelayQueue ) {
        idx -= ldelayQueue;
    }

    // set a spike
    assert( !(delay_queue & (0x1 << idx)) );
    delay_queue |= (0x1 << idx);
}

/**
 *  Fire the selected Neuron and calculate the result.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 *  @param  deltaT  inner simulation step duration
 */
void SingleThreadedSpikingModel::fire(AllNeurons &neurons, const int index, const BGFLOAT deltaT) const
{
    AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons&>(neurons);

    // Note that the neuron has fired!
    spNeurons.hasFired[index] = true;

    // record spike time
    spNeurons.spike_history[index][spNeurons.spikeCount[index]] = g_simulationStep;

    // increment spike count and total spike count
    spNeurons.spikeCount[index]++;
}

/**
 *  Advance all the Synapses in the simulation.
 *  @param  num_neurons number of neurons in the simulation to run.
 *  @param  synapses    list of Synapses to update.
 *  @param  deltaT      inner simulation step duration
 */
void SingleThreadedSpikingModel::advanceSynapses(const int num_neurons, AllSynapses &synapses, const BGFLOAT deltaT)
{
    for (int i = 0; i < num_neurons; i++) {
        size_t synapse_counts = synapses.synapse_counts[i];
        int synapse_advanced = 0;
        for (int z = 0; z < synapse_counts; z++) {
            // Advance Synapse
            uint32_t iSyn = synapses.maxSynapsesPerNeuron * i + z;
            advanceSynapse(synapses, iSyn, deltaT);
            synapse_advanced++;
        }
    }
}

/**
 *  Advance one specific Synapse.
 *  @param  synapses    list of the Synapses to advance.
 *  @param  iSyn   index of the Synapse to connect to.
 *  @param  deltaT   inner simulation step duration
 */
void SingleThreadedSpikingModel::advanceSynapse(AllSynapses &synapses, const uint32_t iSyn, const BGFLOAT deltaT)
{
    uint64_t &lastSpike = synapses.lastSpike[iSyn];
    AllDSSynapses &DsSynapses = dynamic_cast<AllDSSynapses&>(synapses);
    BGFLOAT &r = DsSynapses.r[iSyn];
    BGFLOAT &u = DsSynapses.u[iSyn];
    BGFLOAT &D = DsSynapses.D[iSyn];
    BGFLOAT &F = DsSynapses.F[iSyn];
    BGFLOAT &U = DsSynapses.U[iSyn];
    BGFLOAT &W = synapses.W[iSyn];
    BGFLOAT &decay = synapses.decay[iSyn];
    BGFLOAT &psr = synapses.psr[iSyn];
    BGFLOAT &summationPoint = *(synapses.summationPoint[iSyn]);

    // is an input in the queue?
    if (isSpikeQueue(synapses, iSyn)) {
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
 *  @param  iSyn   index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool SingleThreadedSpikingModel::isSpikeQueue(AllSynapses &synapses, const uint32_t iSyn)
{
    uint32_t &delay_queue = synapses.delayQueue[iSyn];
    int &delayIdx = synapses.delayIdx[iSyn];
    int &ldelayQueue = synapses.ldelayQueue[iSyn];

    bool r = delay_queue & (0x1 << delayIdx);
    delay_queue &= ~(0x1 << delayIdx);
    if ( ++delayIdx >= ldelayQueue ) {
        delayIdx = 0;
    }
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
void SingleThreadedSpikingModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
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
                uint32_t iSyn = synapses.maxSynapsesPerNeuron * src_neuron + synapse_index;
                if (synapses.in_use[iSyn] == true) {
                    // if there is a synapse between a and b
                    if (synapses.summationCoord[iSyn] == dest_coord) {
                        connected = true;
                        adjusted++;

                        // adjust the strength of the synapse or remove
                        // it from the synapse map if it has gone below
                        // zero.
                        if ((*m_conns->W)(src_neuron, dest_neuron) < 0) {
                            removed++;
                            eraseSynapse(synapses, src_neuron, iSyn);
                        } else {
                            // adjust
                            // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                            synapses.W[iSyn] = (*m_conns->W)(src_neuron, dest_neuron) *
                                synSign(type) * AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

                            DEBUG_MID(cout << "weight of rgSynapseMap" <<
                                   coordToString(xa, ya)<<"[" <<synapse_index<<"]: " <<
                                   synapses.W[iSyn] << endl;);
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
 *  @param  iSyn      Index of a synapse.
 */
void SingleThreadedSpikingModel::eraseSynapse(AllSynapses &synapses, const int neuron_index, const uint32_t iSyn)
{
    synapses.synapse_counts[neuron_index]--;
    synapses.in_use[iSyn] = false;
    synapses.summationPoint[iSyn] = NULL;
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
void SingleThreadedSpikingModel::addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, const BGFLOAT deltaT)
{
    if (synapses.synapse_counts[src_neuron] >= synapses.maxSynapsesPerNeuron) {
        return; // TODO: ERROR!
    }

    // add it to the list
    size_t synapse_index;
    uint32_t iSyn;
    for (synapse_index = 0; synapse_index < synapses.maxSynapsesPerNeuron; synapse_index++) {
        iSyn = synapses.maxSynapsesPerNeuron * src_neuron + synapse_index;
        if (!synapses.in_use[iSyn]) {
            break;
        }
    }

    synapses.synapse_counts[src_neuron]++;

    // create a synapse
    synapses.createSynapse(iSyn, source, dest, sum_point, deltaT, type );
    synapses.W[iSyn] = (*m_conns->W)(src_neuron, dest_neuron) 
            * synSign(type) * AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}

/**
 *  TODO
 *  @param  neurons the Neuron list to search from.
 *  @param  src_coord   the coordinate of the source Neuron.
 *  @param  dest_coord  the coordinate of the destination Neuron.
 *  @param  width   TODO
 */
synapseType SingleThreadedSpikingModel::synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const int width)
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
synapseType SingleThreadedSpikingModel::synType(AllNeurons &neurons, const int src_neuron, const int dest_neuron)
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
int SingleThreadedSpikingModel::synSign(const synapseType type)
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
