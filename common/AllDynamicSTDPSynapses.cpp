#include "AllDynamicSTDPSynapses.h"
#include "AllNeurons.h"

AllDynamicSTDPSynapses::AllDynamicSTDPSynapses() : AllSTDPSynapses()
{
    lastSpike = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
}

AllDynamicSTDPSynapses::AllDynamicSTDPSynapses(const int num_neurons, const int max_synapses) :
        AllSTDPSynapses(num_neurons, max_synapses)
{
    setupSynapses(num_neurons, max_synapses);
}

AllDynamicSTDPSynapses::~AllDynamicSTDPSynapses()
{
    cleanupSynapses();
}

void AllDynamicSTDPSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

void AllDynamicSTDPSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    AllSTDPSynapses::setupSynapses(num_neurons, max_synapses);

    uint32_t max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        lastSpike = new uint64_t[max_total_synapses];
        r = new BGFLOAT[max_total_synapses];
        u = new BGFLOAT[max_total_synapses];
        D = new BGFLOAT[max_total_synapses];
        U = new BGFLOAT[max_total_synapses];
        F = new BGFLOAT[max_total_synapses];
    }
}

void AllDynamicSTDPSynapses::cleanupSynapses()
{
    uint32_t max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] lastSpike;
        delete[] r;
        delete[] u;
        delete[] D;
        delete[] U;
        delete[] F;
    }

    lastSpike = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;

    AllSTDPSynapses::cleanupSynapses();
}

/*
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  iSyn   index of the synapse to set.
 */
void AllDynamicSTDPSynapses::readSynapse(istream &input, const uint32_t iSyn)
{
    AllSTDPSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> lastSpike[iSyn]; input.ignore();
    input >> r[iSyn]; input.ignore();
    input >> u[iSyn]; input.ignore();
    input >> D[iSyn]; input.ignore();
    input >> U[iSyn]; input.ignore();
    input >> F[iSyn]; input.ignore();
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  iSyn   index of the synapse to print out.
 */
void AllDynamicSTDPSynapses::writeSynapse(ostream& output, const uint32_t iSyn) const 
{
    AllSTDPSynapses::writeSynapse(output, iSyn);

    output << lastSpike[iSyn] << ends;
    output << r[iSyn] << ends;
    output << u[iSyn] << ends;
    output << D[iSyn] << ends;
    output << U[iSyn] << ends;
    output << F[iSyn] << ends;
}

/**
 *  Reset time varying state vars and recompute decay.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration
 */
void AllDynamicSTDPSynapses::resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT)
{
    AllSTDPSynapses::resetSynapse(iSyn, deltaT);

    u[iSyn] = DEFAULT_U;
    r[iSyn] = 1.0;
    lastSpike[iSyn] = ULONG_MAX;
}

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
void AllDynamicSTDPSynapses::createSynapse(const uint32_t iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllSTDPSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

    U[iSyn] = DEFAULT_U;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            break;
        default:
            assert( false );
            break;
    }

    this->U[iSyn] = U;
    this->D[iSyn] = D;
    this->F[iSyn] = F;
}

#if !defined(USE_GPU)
void AllDynamicSTDPSynapses::changePSR(const uint32_t iSyn, const BGFLOAT deltaT)
{
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &W = this->W[iSyn];
    BGFLOAT &decay = this->decay[iSyn];
    uint64_t &lastSpike = this->lastSpike[iSyn];
    BGFLOAT &r = this->r[iSyn];
    BGFLOAT &u = this->u[iSyn];
    BGFLOAT &D = this->D[iSyn];
    BGFLOAT &F = this->F[iSyn];
    BGFLOAT &U = this->U[iSyn];

    // adjust synapse parameters
    if (lastSpike != ULONG_MAX) {
        BGFLOAT isi = (g_simulationStep - lastSpike) * deltaT ;
        r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
        u = U + u * ( 1 - U ) * exp( -isi / F );
    }
    psr += ( ( W / decay ) * u * r );    // calculate psr
    lastSpike = g_simulationStep;        // record the time of the spike
}

#endif // !defined(USE_GPU)
