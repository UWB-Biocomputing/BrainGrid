#include "AllSpikingSynapses.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif // USE_GPU

// Default constructor
AllSpikingSynapses::AllSpikingSynapses() : AllSynapses()
{
}

// Copy constructor
AllSpikingSynapses::AllSpikingSynapses(const AllSpikingSynapses &r_synapses) : AllSynapses(r_synapses)
{
    copyParameters(dynamic_cast<const AllSpikingSynapses &>(r_synapses));
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
    setupSynapsesInternalState(sim_info, clr_info);

    // allocate synspses properties data
    m_pSynapsesProperties = new AllSpikingSynapsesProperties();
    m_pSynapsesProperties->setupSynapsesProperties(num_neurons, max_synapses, sim_info, clr_info);
}

/*
 *  Setup the internal structure of the class.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapses::setupSynapsesInternalState(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSynapses::setupSynapsesInternalState(sim_info, clr_info);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingSynapses::cleanupSynapses()
{
    // deallocate neurons properties data
    delete m_pSynapsesProperties;
    m_pSynapsesProperties = NULL;

    cleanupSynapsesInternalState();
}

/*
 *  Deallocate all resources.
 */
void AllSpikingSynapses::cleanupSynapsesInternalState()
{
    AllSynapses::cleanupSynapsesInternalState();
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
    dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties)->preSpikeQueue->deserialize(input);
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
    dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties)->preSpikeQueue->serialize(output);
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSpikingSynapses::readSynapse(istream &input, const BGSIZE iSyn)
{
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

    AllSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> pSynapsesProperties->decay[iSyn]; input.ignore();
    input >> pSynapsesProperties->total_delay[iSyn]; input.ignore();
    input >> pSynapsesProperties->tau[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSpikingSynapses::writeSynapse(ostream& output, const BGSIZE iSyn) const
{
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

    AllSynapses::writeSynapse(output, iSyn);

    output << pSynapsesProperties->decay[iSyn] << ends;
    output << pSynapsesProperties->total_delay[iSyn] << ends;
    output << pSynapsesProperties->tau[iSyn] << ends;
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
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);
    BGFLOAT delay;

    pSynapsesProperties->in_use[iSyn] = true;
    pSynapsesProperties->summationPoint[iSyn] = sum_point;
    pSynapsesProperties->destNeuronLayoutIndex[iSyn] = dest_index;
    pSynapsesProperties->sourceNeuronLayoutIndex[iSyn] = source_index;
    pSynapsesProperties->W[iSyn] = synSign(type) * 10.0e-9;
    pSynapsesProperties->type[iSyn] = type;
    pSynapsesProperties->tau[iSyn] = DEFAULT_tau;

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

    pSynapsesProperties->tau[iSyn] = tau;
    pSynapsesProperties->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    assert( pSynapsesProperties->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY );

    // initializes the queues for the Synapses
    pSynapsesProperties->preSpikeQueue->clearAnEvent(iSyn);

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
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);
    int &total_delay = pSynapsesProperties->total_delay[iSyn];

    // Checks if there is an event in the queue.
    return pSynapsesProperties->preSpikeQueue->checkAnEvent(iSyn, total_delay, iStepOffset);
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
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

    // Add to spike queue
    pSynapsesProperties->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
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
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

    pSynapsesProperties->preSpikeQueue->advanceEventQueue(iStep);
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
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

    BGFLOAT &decay = pSynapsesProperties->decay[iSyn];
    BGFLOAT &psr = pSynapsesProperties->psr[iSyn];
    BGFLOAT &summationPoint = *(pSynapsesProperties->summationPoint[iSyn]);

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
    AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

    BGFLOAT &psr = pSynapsesProperties->psr[iSyn];
    BGFLOAT &W = pSynapsesProperties->W[iSyn];
    BGFLOAT &decay = pSynapsesProperties->decay[iSyn];

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
        AllSpikingSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSpikingSynapsesProperties*>(m_pSynapsesProperties);

        BGFLOAT &tau = pSynapsesProperties->tau[iSyn];
        BGFLOAT &decay = pSynapsesProperties->decay[iSyn];

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
