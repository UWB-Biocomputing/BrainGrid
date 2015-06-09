#include "AllSTDPSynapses.h"
#include "AllNeurons.h"

AllSTDPSynapses::AllSTDPSynapses() : AllSpikingSynapses()
{
    total_delayPost = NULL;
    delayQueuePost = NULL;
    delayIdxPost = NULL;
    ldelayQueuePost = NULL;
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

void AllSTDPSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

void AllSTDPSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    AllSpikingSynapses::setupSynapses(num_neurons, max_synapses);

    uint32_t max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        total_delayPost = new int[max_total_synapses];
        delayQueuePost = new uint32_t[max_total_synapses];
        delayIdxPost = new int[max_total_synapses];
        ldelayQueuePost = new int[max_total_synapses];
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
    }
}

void AllSTDPSynapses::cleanupSynapses()
{
    uint32_t max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] total_delayPost;
        delete[] delayQueuePost;
        delete[] delayIdxPost;
        delete[] ldelayQueuePost;
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
    }

    total_delayPost = NULL;
    delayQueuePost = NULL;
    delayIdxPost = NULL;
    ldelayQueuePost = NULL;
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

    AllSpikingSynapses::cleanupSynapses();
}

/**
 *  Initializes the queues for the Synapses.
 *  @param  iSyn   index of the synapse to set.
 */
void AllSTDPSynapses::initSpikeQueue(const uint32_t iSyn)
{
    AllSpikingSynapses::initSpikeQueue(iSyn);

    int &total_delay = this->total_delayPost[iSyn];
    uint32_t &delayQueue = this->delayQueuePost[iSyn];
    int &delayIdx = this->delayIdxPost[iSyn];
    int &ldelayQueue = this->ldelayQueuePost[iSyn];

    size_t size = total_delay / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
    delayQueue = 0;
    delayIdx = 0;
    ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

/*
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  iSyn   index of the synapse to set.
 */
void AllSTDPSynapses::readSynapse(istream &input, const uint32_t iSyn)
{
    AllSpikingSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> total_delayPost[iSyn]; input.ignore();
    input >> delayQueuePost[iSyn]; input.ignore();
    input >> delayIdxPost[iSyn]; input.ignore();
    input >> ldelayQueuePost[iSyn]; input.ignore();
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
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  iSyn   index of the synapse to print out.
 */
void AllSTDPSynapses::writeSynapse(ostream& output, const uint32_t iSyn) const 
{
    AllSpikingSynapses::writeSynapse(output, iSyn);

    output << total_delayPost[iSyn] << ends;
    output << delayQueuePost[iSyn] << ends;
    output << delayIdxPost[iSyn] << ends;
    output << ldelayQueuePost[iSyn] << ends;
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
}

/**
 *  Reset time varying state vars and recompute decay.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration
 */
void AllSTDPSynapses::resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT)
{
    AllSpikingSynapses::resetSynapse(iSyn, deltaT);
}

#if !defined(USE_GPU)
/**
 *  Create a Synapse and connect it to the model.
 *  @param  synapses    the Neuron list to reference.
 *  @param  iSyn   TODO
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param  deltaT  TODO
 *  @param  type    type of the Synapse to create.
 */
void AllSTDPSynapses::createSynapse(const uint32_t iSyn, Coordinate source, Coordinate dest, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllSpikingSynapses::createSynapse(iSyn, source, dest, sum_point, deltaT, type);

    Apos[iSyn] = 0.5;
    Aneg[iSyn] = -0.5;
    STDPgap[iSyn] = 2e-3;

/* TODO: these values need to be initialized
    total_delayPost = 0;
    delayQueuePost = 0;
    delayIdxPost = 0;
    ldelayQueuePost = 0;
    tauspost = 0;
    tauspre = 0;
    taupos = 0;
    tauneg = 0;
    STDPgap = 0;
    Wex = 0;
    Aneg = 0;
    Apos = 0;
    mupos = 0;
    muneg = 0;
*/
}

/**
 *  Advance one specific Synapse.
 *  @param  iSyn   index of the Synapse to connect to.
 *  @param  deltaT   inner simulation step duration
 */
void AllSTDPSynapses::advanceSynapse(const uint32_t iSyn, const SimulationInfo *sim_info, AllNeurons *neurons)
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

        BGFLOAT deltaT = sim_info->deltaT;
        AllSpikingNeurons* spNeurons = dynamic_cast<AllSpikingNeurons*>(neurons);

        // pre and post neurons index
        int idxPre = iSyn / maxSynapsesPerNeuron; 
        int idxPost = summationCoord[iSyn].x + summationCoord[iSyn].y * sim_info->width;
        uint64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {	// preSpikeHit
            spikeHistory = spNeurons->getSpikeHistory(idxPre, -2, sim_info);
            if (spikeHistory > 0) {
                delta = (g_simulationStep - spikeHistory) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
            } else {
                epre = 1.0;
            }

            int offIndex = -1;
            while (true) {
                spikeHistory = spNeurons->getSpikeHistory(idxPost, offIndex, sim_info);
                if (spikeHistory <= 0)
                    break;
                delta = (spikeHistory - g_simulationStep) * deltaT;
                if (delta <= -3.0 * tauneg)
                    break;
                spikeHistory2 = spNeurons->getSpikeHistory(idxPost, offIndex-1, sim_info);
                if (spikeHistory2 <= 0)
                    break;
                delta = (spikeHistory - spikeHistory2) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
                stdpLearning(iSyn, delta, epost, epre);
                --offIndex;
            }

            changePSR(iSyn, deltaT);
        }

        if (fPost) {	// postSpikeHit
            spikeHistory = spNeurons->getSpikeHistory(idxPost, -2, sim_info);
            if (spikeHistory > 0) {
                delta = (g_simulationStep - spikeHistory) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
            } else {
                epost = 1.0;
            }

            int offIndex = -1;
            while (true) {
                spikeHistory = spNeurons->getSpikeHistory(idxPre, offIndex, sim_info);
                if (spikeHistory <= 0)
                    break;
                delta = (spikeHistory - g_simulationStep) * deltaT - total_delay;
                if (delta <= 0 || delta >= 3.0 * taupos)
                    break;
                spikeHistory2 = spNeurons->getSpikeHistory(idxPre, offIndex-1, sim_info);
                if (spikeHistory2 <= 0)
                    break;                
                delta = (spikeHistory - spikeHistory2) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
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

void AllSTDPSynapses::stdpLearning(const uint32_t iSyn, double delta, double epost, double epre)
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
}

/**
 *  Checks if there is an input spike in the queue.
 *  @param  iSyn   index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool AllSTDPSynapses::isSpikeQueuePost(const uint32_t iSyn)
{
    uint32_t &delayQueue = this->delayQueuePost[iSyn];
    int &delayIdx = this->delayIdxPost[iSyn];
    int &ldelayQueue = this->ldelayQueuePost[iSyn];

    bool r = delayQueue & (0x1 << delayIdx);
    delayQueue &= ~(0x1 << delayIdx);
    if ( ++delayIdx >= ldelayQueue ) {
        delayIdx = 0;
    }
    return r;
}

void AllSTDPSynapses::postSpikeHit(const uint32_t iSyn)
{
    uint32_t &delay_queue = this->delayQueuePost[iSyn];
    int &delayIdx = this->delayIdxPost[iSyn];
    int &ldelayQueue = this->ldelayQueuePost[iSyn];
    int &total_delay = this->total_delayPost[iSyn];

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

bool AllSTDPSynapses::allowBackPropagation()
{
    return true;
}
#endif // !defined(USE_GPU)
