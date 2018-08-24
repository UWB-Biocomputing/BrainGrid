#include "AllSpikingSynapses.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif // USE_GPU

// Default constructor
AllSpikingSynapses::AllSpikingSynapses() : AllSynapses()
{
    decay = NULL;
    total_delay = NULL;
    tau = NULL;
    preSpikeQueue = NULL;
}

// Copy constructor
AllSpikingSynapses::AllSpikingSynapses(const AllSpikingSynapses &r_synapses) : AllSynapses(r_synapses)
{
    decay = NULL;
    total_delay = NULL;
    tau = NULL;
    preSpikeQueue = NULL;
}

AllSpikingSynapses::~AllSpikingSynapses()
{
    cleanupSynapses();
}

/*
 *  Assignment operator: copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
IAllSynapses &AllSpikingSynapses::operator=(const IAllSynapses &r_synapses)
{
    copyParameters(dynamic_cast<const AllSpikingSynapses &>(r_synapses));

    return (*this);
}

/*
 *  Copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
void AllSpikingSynapses::copyParameters(const AllSpikingSynapses &r_synapses)
{
    AllSynapses::copyParameters(r_synapses);
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapses::setupSynapses(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    setupSynapses(clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron, sim_info, clr_info);
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 *  @param  sim_info      SimulationInfo class to read information from.
 *  @param  clr_info      ClusterInfo class to read information from.
 */
void AllSpikingSynapses::setupSynapses(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSynapses::setupSynapses(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        decay = new BGFLOAT[max_total_synapses];
        total_delay = new int[max_total_synapses];
        tau = new BGFLOAT[max_total_synapses];

        // create a pre synapse spike queue & initialize it
        preSpikeQueue = new EventQueue();
#if defined(USE_GPU)
        // max_total_synapses * sim_info->maxFiringRate may overflow the maximum range
        // of 32 bits integer, so we cast it uint64_t
        int nMaxInterClustersOutgoingEvents = (uint64_t) max_total_synapses * sim_info->maxFiringRate * sim_info->deltaT * sim_info->minSynapticTransDelay;
        int nMaxInterClustersIncomingEvents = (uint64_t) max_total_synapses * sim_info->maxFiringRate * sim_info->deltaT * sim_info->minSynapticTransDelay;

        // initializes the pre synapse spike queue
        preSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses, nMaxInterClustersOutgoingEvents, nMaxInterClustersIncomingEvents);
#else // USE_GPU
        // initializes the pre synapse spike queue
        preSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses);
#endif // USE_GPU

        // register the queue to the event handler
        if (clr_info->eventHandler != NULL) {
            clr_info->eventHandler->addEventQueue(clr_info->clusterID, preSpikeQueue);
        }
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingSynapses::cleanupSynapses()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] decay;
        delete[] total_delay;
        delete[] tau;
    }

    decay = NULL;
    total_delay = NULL;
    tau = NULL;

    if (preSpikeQueue != NULL) {
        delete preSpikeQueue;
        preSpikeQueue = NULL;
    }

    AllSynapses::cleanupSynapses();
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
void AllSpikingSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllSynapses::resetSynapse(iSyn, deltaT);

    assert( updateDecay(iSyn, deltaT) );
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllSpikingSynapses::checkNumParameters()
{
    return (nParams >= 0);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllSpikingSynapses::readParameters(const TiXmlElement& element)
{
    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllSpikingSynapses::printParameters(ostream &output) const
{
}

/*
 *  Sets the data for Synapses to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapses::deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info)
{
    AllSynapses::deserialize(input, neurons, clr_info);
    preSpikeQueue->deserialize(input);
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapses::serialize(ostream& output, const ClusterInfo *clr_info)
{
    AllSynapses::serialize(output, clr_info);
    preSpikeQueue->serialize(output);
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSpikingSynapses::readSynapse(istream &input, const BGSIZE iSyn)
{
    AllSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> decay[iSyn]; input.ignore();
    input >> total_delay[iSyn]; input.ignore();
    input >> tau[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSpikingSynapses::writeSynapse(ostream& output, const BGSIZE iSyn) const
{
    AllSynapses::writeSynapse(output, iSyn);

    output << decay[iSyn] << ends;
    output << total_delay[iSyn] << ends;
    output << tau[iSyn] << ends;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  synapses    The synapse list to reference.
 *  @param  iSyn        Index of the synapse to set.
 *  @param  source      Coordinates of the source Neuron.
 *  @param  dest        Coordinates of the destination Neuron.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllSpikingSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;

    in_use[iSyn] = true;
    summationPoint[iSyn] = sum_point;
    destNeuronLayoutIndex[iSyn] = dest_index;
    sourceNeuronLayoutIndex[iSyn] = source_index;
    W[iSyn] = synSign(type) * 10.0e-9;
    this->type[iSyn] = type;
    tau[iSyn] = DEFAULT_tau;

    BGFLOAT tau;
    switch (type) {
        case II:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            assert( false );
            break;
    }

    this->tau[iSyn] = tau;
    this->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    // initializes the queues for the Synapses
    preSpikeQueue->clearAnEvent(iSyn);

    // reset time varying state vars and recompute decay
    resetSynapse(iSyn, deltaT);
}

#if !defined(USE_GPU)
/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @param  iStepOffset  Offset from the current simulation step.
 *  @return true if there is an input spike event.
 */
bool AllSpikingSynapses::isSpikeQueue(const BGSIZE iSyn, int iStepOffset)
{
    int &total_delay = this->total_delay[iSyn];

    // Checks if there is an event in the queue.
    return preSpikeQueue->checkAnEvent(iSyn, total_delay, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param  iSyn   Index of the Synapse to update.
 *  @param  iStepOffset  Offset from the current simulation step.
 *  @param  iCluster  Cluster ID of cluster where the spike is added.
 */
void AllSpikingSynapses::preSpikeHit(const BGSIZE iSyn, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset)
{
    // Add to spike queue
    preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to update.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
void AllSpikingSynapses::postSpikeHit(const BGSIZE iSyn, int iStepOffset)
{
}

/*
 *  Advance all the Synapses in the simulation.
 *
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  neurons          The Neuron list to search from.
 *  @param  synapseIndexMap  Pointer to the synapse index map.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSpikingSynapses::advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset)
{
    AllSynapses::advanceSynapses(sim_info, neurons, synapseIndexMap, iStepOffset);
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 * @param iStep     simulation steps to advance.
 */
void AllSpikingSynapses::advanceSpikeQueue(int iStep)
{
    preSpikeQueue->advanceEventQueue(iStep);
}

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
void AllSpikingSynapses::advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons * neurons, int iStepOffset)
{
    BGFLOAT &decay = this->decay[iSyn];
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &summationPoint = *(this->summationPoint[iSyn]);

    // is an input in the queue?
    if (isSpikeQueue(iSyn, iStepOffset)) {
        changePSR(iSyn, sim_info->deltaT, iStepOffset);
    }

    // decay the post spike response
    psr *= decay;
    // and apply it to the summation point
    summationPoint += psr;
}

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSpikingSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, int iStepOffset)
{
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &W = this->W[iSyn];
    BGFLOAT &decay = this->decay[iSyn];

    psr += ( W / decay );    // calculate psr
}

#endif //!defined(USE_GPU)

/*
 *  Updates the decay if the synapse selected.
 *
 *  @param  iSyn    Index of the synapse to set.
 *  @param  deltaT  Inner simulation step duration
 */
bool AllSpikingSynapses::updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT)
{
        BGFLOAT &tau = this->tau[iSyn];
        BGFLOAT &decay = this->decay[iSyn];

        if (tau > 0) {
                decay = exp( -deltaT / tau );
                return true;
        }
        return false;
}

/*
 *  Check if the back propagation (notify a spike event to the pre neuron)
 *  is allowed in the synapse class.
 *
 *  @retrun true if the back propagation is allowed.
 */
bool AllSpikingSynapses::allowBackPropagation()
{
    return false;
}
