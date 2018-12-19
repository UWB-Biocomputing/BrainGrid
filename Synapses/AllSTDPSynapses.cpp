#include "AllSTDPSynapses.h"
#include "IAllNeurons.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif // USE_GPU

// Default constructor
AllSTDPSynapses::AllSTDPSynapses() : AllSpikingSynapses()
{
}

// Copy constructor
AllSTDPSynapses::AllSTDPSynapses(const AllSTDPSynapses &r_synapses) : AllSpikingSynapses(r_synapses)
{
    copyParameters(dynamic_cast<const AllSTDPSynapses &>(r_synapses));
}

AllSTDPSynapses::~AllSTDPSynapses()
{
    cleanupSynapses();
}

/*
 *  Assignment operator: copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
IAllSynapses &AllSTDPSynapses::operator=(const IAllSynapses &r_synapses)
{
    copyParameters(dynamic_cast<const AllSTDPSynapses &>(r_synapses));

    return (*this);
}

/*
 *  Copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
void AllSTDPSynapses::copyParameters(const AllSTDPSynapses &r_synapses)
{
    AllSpikingSynapses::copyParameters(r_synapses);
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSTDPSynapses::setupSynapses(SimulationInfo *sim_info, ClusterInfo *clr_info)
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
void AllSTDPSynapses::setupSynapses(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    setupSynapsesInternalState(sim_info, clr_info);

    // allocate synspses properties data
    m_pSynapsesProperties = new AllSTDPSynapsesProperties();
    m_pSynapsesProperties->setupSynapsesProperties(num_neurons, max_synapses, sim_info, clr_info);
}

/*
 *  Setup the internal structure of the class.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSTDPSynapses::setupSynapsesInternalState(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingSynapses::setupSynapsesInternalState(sim_info, clr_info);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSTDPSynapses::cleanupSynapses()
{
     // deallocate neurons properties data
    delete m_pSynapsesProperties;
    m_pSynapsesProperties = NULL;

    cleanupSynapsesInternalState();
}

/*
 *  Deallocate all resources.
 */
void AllSTDPSynapses::cleanupSynapsesInternalState()
{
    AllSpikingSynapses::cleanupSynapsesInternalState();
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllSTDPSynapses::checkNumParameters()
{
    return (nParams >= 0);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllSTDPSynapses::readParameters(const TiXmlElement& element)
{
    if (AllSpikingSynapses::readParameters(element)) {
        // this parameter was already handled
        return true;
    }

    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllSTDPSynapses::printParameters(ostream &output) const
{
}

/*
 *  Sets the data for Synapses to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSTDPSynapses::deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    AllSpikingSynapses::deserialize(input, neurons, clr_info);
    pSynapsesProperties->postSpikeQueue->deserialize(input);
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSTDPSynapses::serialize(ostream& output, const ClusterInfo *clr_info)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    AllSpikingSynapses::serialize(output, clr_info);
    pSynapsesProperties->postSpikeQueue->serialize(output);
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSTDPSynapses::readSynapse(istream &input, const BGSIZE iSyn)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);
 
    AllSpikingSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> pSynapsesProperties->total_delayPost[iSyn]; input.ignore();
    input >> pSynapsesProperties->tauspost[iSyn]; input.ignore();
    input >> pSynapsesProperties->tauspre[iSyn]; input.ignore();
    input >> pSynapsesProperties->taupos[iSyn]; input.ignore();
    input >> pSynapsesProperties->tauneg[iSyn]; input.ignore();
    input >> pSynapsesProperties->STDPgap[iSyn]; input.ignore();
    input >> pSynapsesProperties->Wex[iSyn]; input.ignore();
    input >> pSynapsesProperties->Aneg[iSyn]; input.ignore();
    input >> pSynapsesProperties->Apos[iSyn]; input.ignore();
    input >> pSynapsesProperties->mupos[iSyn]; input.ignore();
    input >> pSynapsesProperties->muneg[iSyn]; input.ignore();
    input >> pSynapsesProperties->useFroemkeDanSTDP[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSTDPSynapses::writeSynapse(ostream& output, const BGSIZE iSyn) const 
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    AllSpikingSynapses::writeSynapse(output, iSyn);

    output << pSynapsesProperties->total_delayPost[iSyn] << ends;
    output << pSynapsesProperties->tauspost[iSyn] << ends;
    output << pSynapsesProperties->tauspre[iSyn] << ends;
    output << pSynapsesProperties->taupos[iSyn] << ends;
    output << pSynapsesProperties->tauneg[iSyn] << ends;
    output << pSynapsesProperties->STDPgap[iSyn] << ends;
    output << pSynapsesProperties->Wex[iSyn] << ends;
    output << pSynapsesProperties->Aneg[iSyn] << ends;
    output << pSynapsesProperties->Apos[iSyn] << ends;
    output << pSynapsesProperties->mupos[iSyn] << ends;
    output << pSynapsesProperties->muneg[iSyn] << ends;
    output << pSynapsesProperties->useFroemkeDanSTDP[iSyn] << ends;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
void AllSTDPSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllSpikingSynapses::resetSynapse(iSyn, deltaT);
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
void AllSTDPSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    AllSpikingSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

    pSynapsesProperties->Apos[iSyn] = 0.5;
    pSynapsesProperties->Aneg[iSyn] = -0.5;
    pSynapsesProperties->STDPgap[iSyn] = 2e-3;

    pSynapsesProperties->total_delayPost[iSyn] = 0;

    pSynapsesProperties->tauspost[iSyn] = 75e-3;
    pSynapsesProperties->tauspre[iSyn] = 34e-3;

    pSynapsesProperties->taupos[iSyn] = 15e-3;
    pSynapsesProperties->tauneg[iSyn] = 35e-3;
    pSynapsesProperties->Wex[iSyn] = 1.0;

    pSynapsesProperties->mupos[iSyn] = 0;
    pSynapsesProperties->muneg[iSyn] = 0;

    pSynapsesProperties->useFroemkeDanSTDP[iSyn] = true;

    // initializes the queues for the Synapses
    pSynapsesProperties->postSpikeQueue->clearAnEvent(iSyn);
}

#if !defined(USE_GPU)
/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
void AllSTDPSynapses::advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons *neurons, int iStepOffset)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    BGFLOAT &decay = pSynapsesProperties->decay[iSyn];
    BGFLOAT &psr = pSynapsesProperties->psr[iSyn];
    BGFLOAT &summationPoint = *(pSynapsesProperties->summationPoint[iSyn]);

    // is an input in the queue?
    bool fPre = isSpikeQueue(iSyn, iStepOffset); 
    bool fPost = isSpikeQueuePost(iSyn, iStepOffset);
    if (fPre || fPost) {
        BGFLOAT &tauspre = pSynapsesProperties->tauspre[iSyn];
        BGFLOAT &tauspost = pSynapsesProperties->tauspost[iSyn];
        BGFLOAT &taupos = pSynapsesProperties->taupos[iSyn];
        BGFLOAT &tauneg = pSynapsesProperties->tauneg[iSyn];
        int &total_delay = pSynapsesProperties->total_delay[iSyn];
        bool &useFroemkeDanSTDP = pSynapsesProperties->useFroemkeDanSTDP[iSyn];

        BGFLOAT deltaT = sim_info->deltaT;
        AllSpikingNeurons* spNeurons = dynamic_cast<AllSpikingNeurons*>(neurons);

        // pre and post neurons index
        int idxPre = pSynapsesProperties->sourceNeuronLayoutIndex[iSyn];
        int idxPost = pSynapsesProperties->destNeuronLayoutIndex[iSyn];
        uint64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {	// preSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time 
            // just one before the last spike.
            spikeHistory = spNeurons->getSpikeHistory(idxPre, -2, sim_info);
            if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)g_simulationStep + iStepOffset - spikeHistory) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
            } else {
                epre = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // pre-post spikes
            int offIndex = -1;	// last spike
            while (true) {
                spikeHistory = spNeurons->getSpikeHistory(idxPost, offIndex, sim_info);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between pre-post spikes
                // (include pre-synaptic transmission delay)
                delta = (spikeHistory - (int64_t)g_simulationStep + iStepOffset) * deltaT;

                DEBUG_SYNAPSE(
                    cout << "AllSTDPSynapses::advanceSynapse: fPre" << endl;
                    cout << "          iSyn: " << iSyn << endl;
                    cout << "          idxPre: " << idxPre << endl;
                    cout << "          idxPost: " << idxPost << endl;
                    cout << "          spikeHistory: " << spikeHistory << endl;
                    cout << "          simulationStep: " << g_simulationStep + iStepOffset << endl;
                    cout << "          delta: " << delta << endl << endl;
                );

                if (delta <= -3.0 * tauneg)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = spNeurons->getSpikeHistory(idxPost, offIndex-1, sim_info);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspost);
                } else {
                    epost = 1.0;
                }
                stdpLearning(iSyn, delta, epost, epre);
                --offIndex;
            }

            changePSR(iSyn, deltaT, iStepOffset);
        }

        if (fPost) {	// postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = spNeurons->getSpikeHistory(idxPost, -2, sim_info);
            if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)g_simulationStep + iStepOffset - spikeHistory) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
            } else {
                epost = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // post-pre spikes
            int offIndex = -1;	// last spike
            while (true) {
                spikeHistory = spNeurons->getSpikeHistory(idxPre, offIndex, sim_info);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between post-pre spikes
                delta = ((int64_t)g_simulationStep + iStepOffset - spikeHistory - total_delay) * deltaT;

                DEBUG_SYNAPSE(
                    cout << "AllSTDPSynapses::advanceSynapse: fPost" << endl;
                    cout << "          iSyn: " << iSyn << endl;
                    cout << "          idxPre: " << idxPre << endl;
                    cout << "          idxPost: " << idxPost << endl;
                    cout << "          spikeHistory: " << spikeHistory << endl;
                    cout << "          simulationStep: " << g_simulationStep + iStepOffset << endl;
                    cout << "          delta: " << delta << endl << endl;
                );

                if (delta <= 0 || delta >= 3.0 * taupos)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = spNeurons->getSpikeHistory(idxPre, offIndex-1, sim_info);
                    if (spikeHistory2 == ULONG_MAX)
                        break;                
                    epre = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspre);
                } else {
                    epre = 1.0;
                }
                stdpLearning(iSyn, delta, epost, epre);
                --offIndex;
            }
        }
    }

    // decay the post spike response
    psr *= decay;
    // and apply it to the summation point
    summationPoint += psr;
}

/*
 *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification 
 *  induced by natural spike trains
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  delta       Pre/post synaptic spike interval.
 *  @param  epost       Params for the rule given in Froemke and Dan (2002).
 *  @param  epre        Params for the rule given in Froemke and Dan (2002).
 */
void AllSTDPSynapses::stdpLearning(const BGSIZE iSyn, double delta, double epost, double epre)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    BGFLOAT STDPgap = pSynapsesProperties->STDPgap[iSyn];
    BGFLOAT muneg = pSynapsesProperties->muneg[iSyn];
    BGFLOAT mupos = pSynapsesProperties->mupos[iSyn];
    BGFLOAT tauneg = pSynapsesProperties->tauneg[iSyn];
    BGFLOAT taupos = pSynapsesProperties->taupos[iSyn];
    BGFLOAT Aneg = pSynapsesProperties->Aneg[iSyn];
    BGFLOAT Apos = pSynapsesProperties->Apos[iSyn];
    BGFLOAT Wex = pSynapsesProperties->Wex[iSyn];
    BGFLOAT &W = pSynapsesProperties->W[iSyn];
    BGFLOAT dw;

    if (delta < -STDPgap) {
        // Depression
        dw = pow(W, muneg) * Aneg * exp(delta / tauneg);
    } else if (delta > STDPgap) {
        // Potentiation
        dw = pow(Wex - W, mupos) * Apos * exp(-delta / taupos);
    } else {
        return;
    }

    W += epost * epre * dw;

    // check the sign
    if ((Wex < 0 && W > 0) || (Wex > 0 && W < 0)) W = 0;

    // check for greater Wmax
    if (fabs(W) > fabs(Wex)) W = Wex;

    DEBUG_SYNAPSE(
        cout << "AllSTDPSynapses::stdpLearning:" << endl;
        cout << "          iSyn: " << iSyn << endl;
        cout << "          delta: " << delta << endl;
        cout << "          epre: " << epre << endl;
        cout << "          epost: " << epost << endl;
        cout << "          dw: " << dw << endl;
        cout << "          W: " << W << endl << endl;
    );
}

/*
 *  Checks if there is an input spike in the queue (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @param  iStepOffset  Offset from the current simulation step.
 *  @return true if there is an input spike event.
 */
bool AllSTDPSynapses::isSpikeQueuePost(const BGSIZE iSyn, int iStepOffset)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    // Checks if there is an event in the queue
    return pSynapsesProperties->postSpikeQueue->checkAnEvent(iSyn, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
void AllSTDPSynapses::postSpikeHit(const BGSIZE iSyn, int iStepOffset)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    int &total_delay = pSynapsesProperties->total_delayPost[iSyn];

    // Add to spike queue
    pSynapsesProperties->postSpikeQueue->addAnEvent(iSyn, total_delay, iStepOffset);
}

/*
 *  Advance all the Synapses in the simulation.
 *
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  neurons          The Neuron list to search from.
 *  @param  synapseIndexMap  Pointer to the synapse index map.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSTDPSynapses::advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset)
{
    AllSpikingSynapses::advanceSynapses(sim_info, neurons, synapseIndexMap, iStepOffset);
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 * @param iStep     simulation steps to advance.
 */
void AllSTDPSynapses::advanceSpikeQueue(int iStep)
{
    AllSTDPSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSTDPSynapsesProperties*>(m_pSynapsesProperties);

    AllSpikingSynapses::advanceSpikeQueue(iStep);

    pSynapsesProperties->postSpikeQueue->advanceEventQueue(iStep);
}
#endif // !defined(USE_GPU)

/*
 *  Check if the back propagation (notify a spike event to the pre neuron)
 *  is allowed in the synapse class.
 *
 *  @retrun true if the back propagation is allowed.
 */
bool AllSTDPSynapses::allowBackPropagation()
{
    return true;
}

