#include "AllSTDPSynapses.h"
#include "IAllNeurons.h"

AllSTDPSynapses::AllSTDPSynapses() : AllSpikingSynapses()
{
    total_delayPost = NULL;
    tauspost = NULL;
    tauspre = NULL;
    taupos = NULL;
    tauneg = NULL;
    STDPgap = NULL;
    Wex = NULL;
    Aneg = NULL;
    Apos = NULL;
    mupos = NULL;
    muneg = NULL;
    useFroemkeDanSTDP = NULL;
    postSpikeQueue = NULL;
}

AllSTDPSynapses::AllSTDPSynapses(const int num_neurons, const int max_synapses) :
        AllSpikingSynapses(num_neurons, max_synapses)
{
    setupSynapses(num_neurons, max_synapses);
}

AllSTDPSynapses::~AllSTDPSynapses()
{
    cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSTDPSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    AllSpikingSynapses::setupSynapses(num_neurons, max_synapses);

    BGSIZE max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        total_delayPost = new int[max_total_synapses];
        tauspost = new BGFLOAT[max_total_synapses];
        tauspre = new BGFLOAT[max_total_synapses];
        taupos = new BGFLOAT[max_total_synapses];
        tauneg = new BGFLOAT[max_total_synapses];
        STDPgap = new BGFLOAT[max_total_synapses];
        Wex = new BGFLOAT[max_total_synapses];
        Aneg = new BGFLOAT[max_total_synapses];
        Apos = new BGFLOAT[max_total_synapses];
        mupos = new BGFLOAT[max_total_synapses];
        muneg = new BGFLOAT[max_total_synapses];
        useFroemkeDanSTDP = new bool[max_total_synapses];

        // create a post synapse spike queue & initialize it
        postSpikeQueue = new EventQueue();
        postSpikeQueue->initEventQueue(max_total_synapses);

    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSTDPSynapses::cleanupSynapses()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] total_delayPost;
        delete[] tauspost;
        delete[] tauspre;
        delete[] taupos;
        delete[] tauneg;
        delete[] STDPgap;
        delete[] Wex;
        delete[] Aneg;
        delete[] Apos;
        delete[] mupos;
        delete[] muneg;
        delete[] useFroemkeDanSTDP;
    }

    total_delayPost = NULL;
    tauspost = NULL;
    tauspre = NULL;
    taupos = NULL;
    tauneg = NULL;
    STDPgap = NULL;
    Wex = NULL;
    Aneg = NULL;
    Apos = NULL;
    mupos = NULL;
    muneg = NULL;
    useFroemkeDanSTDP = NULL;

    if (postSpikeQueue != NULL) {
        delete[] postSpikeQueue;
        postSpikeQueue = NULL;
    }

    AllSpikingSynapses::cleanupSynapses();
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
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSTDPSynapses::deserialize(istream& input, IAllNeurons &neurons, const SimulationInfo *sim_info)
{
    AllSpikingSynapses::deserialize(input, neurons, sim_info);
    postSpikeQueue->deserialize(input);
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSTDPSynapses::serialize(ostream& output, const SimulationInfo *sim_info)
{
    AllSpikingSynapses::serialize(output, sim_info);
    postSpikeQueue->serialize(output);
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSTDPSynapses::readSynapse(istream &input, const BGSIZE iSyn)
{
    AllSpikingSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> total_delayPost[iSyn]; input.ignore();
    input >> tauspost[iSyn]; input.ignore();
    input >> tauspre[iSyn]; input.ignore();
    input >> taupos[iSyn]; input.ignore();
    input >> tauneg[iSyn]; input.ignore();
    input >> STDPgap[iSyn]; input.ignore();
    input >> Wex[iSyn]; input.ignore();
    input >> Aneg[iSyn]; input.ignore();
    input >> Apos[iSyn]; input.ignore();
    input >> mupos[iSyn]; input.ignore();
    input >> muneg[iSyn]; input.ignore();
    input >> useFroemkeDanSTDP[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSTDPSynapses::writeSynapse(ostream& output, const BGSIZE iSyn) const 
{
    AllSpikingSynapses::writeSynapse(output, iSyn);

    output << total_delayPost[iSyn] << ends;
    output << tauspost[iSyn] << ends;
    output << tauspre[iSyn] << ends;
    output << taupos[iSyn] << ends;
    output << tauneg[iSyn] << ends;
    output << STDPgap[iSyn] << ends;
    output << Wex[iSyn] << ends;
    output << Aneg[iSyn] << ends;
    output << Apos[iSyn] << ends;
    output << mupos[iSyn] << ends;
    output << muneg[iSyn] << ends;
    output << useFroemkeDanSTDP[iSyn] << ends;
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
    AllSpikingSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

    Apos[iSyn] = 0.5;
    Aneg[iSyn] = -0.5;
    STDPgap[iSyn] = 2e-3;

    total_delayPost[iSyn] = 0;

    tauspost[iSyn] = 75e-3;
    tauspre[iSyn] = 34e-3;

    taupos[iSyn] = 15e-3;
    tauneg[iSyn] = 35e-3;
    Wex[iSyn] = 1.0;

    mupos[iSyn] = 0;
    muneg[iSyn] = 0;

    useFroemkeDanSTDP[iSyn] = true;

    // initializes the queues for the Synapses
    postSpikeQueue->clearAnEvent(iSyn);
}

#if !defined(USE_GPU)
/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 */
void AllSTDPSynapses::advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons *neurons)
{
    BGFLOAT &decay = this->decay[iSyn];
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &summationPoint = *(this->summationPoint[iSyn]);

    // is an input in the queue?
    bool fPre = isSpikeQueue(iSyn); 
    bool fPost = isSpikeQueuePost(iSyn);
    if (fPre || fPost) {
        BGFLOAT &tauspre = this->tauspre[iSyn];
        BGFLOAT &tauspost = this->tauspost[iSyn];
        BGFLOAT &taupos = this->taupos[iSyn];
        BGFLOAT &tauneg = this->tauneg[iSyn];
        int &total_delay = this->total_delay[iSyn];
        bool &useFroemkeDanSTDP = this->useFroemkeDanSTDP[iSyn];

        BGFLOAT deltaT = sim_info->deltaT;
        AllSpikingNeurons* spNeurons = dynamic_cast<AllSpikingNeurons*>(neurons);

        // pre and post neurons index
        int idxPre = sourceNeuronIndex[iSyn];
        int idxPost = destNeuronIndex[iSyn];
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
                delta = ((int64_t)g_simulationStep - spikeHistory) * deltaT;
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
                delta = (spikeHistory - (int64_t)g_simulationStep) * deltaT;

                DEBUG_SYNAPSE(
                    cout << "AllSTDPSynapses::advanceSynapse: fPre" << endl;
                    cout << "          iSyn: " << iSyn << endl;
                    cout << "          idxPre: " << idxPre << endl;
                    cout << "          idxPost: " << idxPost << endl;
                    cout << "          spikeHistory: " << spikeHistory << endl;
                    cout << "          g_simulationStep: " << g_simulationStep << endl;
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

            changePSR(iSyn, deltaT);
        }

        if (fPost) {	// postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = spNeurons->getSpikeHistory(idxPost, -2, sim_info);
            if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)g_simulationStep - spikeHistory) * deltaT;
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
                delta = ((int64_t)g_simulationStep - spikeHistory - total_delay) * deltaT;

                DEBUG_SYNAPSE(
                    cout << "AllSTDPSynapses::advanceSynapse: fPost" << endl;
                    cout << "          iSyn: " << iSyn << endl;
                    cout << "          idxPre: " << idxPre << endl;
                    cout << "          idxPost: " << idxPost << endl;
                    cout << "          spikeHistory: " << spikeHistory << endl;
                    cout << "          g_simulationStep: " << g_simulationStep << endl;
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
#ifdef USE_OMP
#pragma omp atomic
#endif
    summationPoint += psr;
#ifdef USE_OMP
    //PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
    //#pragma omp flush (summationPoint)
#endif
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
    BGFLOAT STDPgap = this->STDPgap[iSyn];
    BGFLOAT muneg = this->muneg[iSyn];
    BGFLOAT mupos = this->mupos[iSyn];
    BGFLOAT tauneg = this->tauneg[iSyn];
    BGFLOAT taupos = this->taupos[iSyn];
    BGFLOAT Aneg = this->Aneg[iSyn];
    BGFLOAT Apos = this->Apos[iSyn];
    BGFLOAT Wex = this->Wex[iSyn];
    BGFLOAT &W = this->W[iSyn];
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
 *  @return true if there is an input spike event.
 */
bool AllSTDPSynapses::isSpikeQueuePost(const BGSIZE iSyn)
{
    return postSpikeQueue->checkAnEvent(iSyn);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 */
void AllSTDPSynapses::postSpikeHit(const BGSIZE iSyn)
{
    int &total_delay = this->total_delayPost[iSyn];

    // Add to spike queue
    postSpikeQueue->addAnEvent(iSyn, total_delay);
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

/*
 *  Advance all the Synapses in the simulation.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 */
void AllSTDPSynapses::advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons)
{
    AllSpikingSynapses::advanceSynapses(sim_info, neurons);

    postSpikeQueue->advanceEventQueue();
}
