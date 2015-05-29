#include "AllDSSynapses.h"
#include "AllNeurons.h"

const BGFLOAT AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

AllDSSynapses::AllDSSynapses() : AllSpikingSynapses()
{
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
}

AllDSSynapses::AllDSSynapses(const int num_neurons, const int max_synapses) :
        AllSpikingSynapses(num_neurons, max_synapses)
{
    setupSynapses(num_neurons, max_synapses);
}

AllDSSynapses::~AllDSSynapses()
{
    cleanupSynapses();
}

void AllDSSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

void AllDSSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    AllSpikingSynapses::setupSynapses(num_neurons, max_synapses);

    uint32_t max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        r = new BGFLOAT[max_total_synapses];
        u = new BGFLOAT[max_total_synapses];
        D = new BGFLOAT[max_total_synapses];
        U = new BGFLOAT[max_total_synapses];
        F = new BGFLOAT[max_total_synapses];
    }
}

void AllDSSynapses::cleanupSynapses()
{
    uint32_t max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] r;
        delete[] u;
        delete[] D;
        delete[] U;
        delete[] F;
    }

    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;

    AllSpikingSynapses::cleanupSynapses();
}

/*
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  iSyn   index of the synapse to set.
 */
void AllDSSynapses::readSynapse(istream &input, const uint32_t iSyn)
{
    AllSpikingSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
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
void AllDSSynapses::writeSynapse(ostream& output, const uint32_t iSyn) const 
{
    AllSpikingSynapses::writeSynapse(output, iSyn);

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
void AllDSSynapses::resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT)
{
        psr[iSyn] = 0.0;
        assert( updateDecay(iSyn, deltaT) );
        u[iSyn] = DEFAULT_U;
        r[iSyn] = 1.0;
        lastSpike[iSyn] = ULONG_MAX;
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
void AllDSSynapses::createSynapse(const uint32_t iSyn, Coordinate source, Coordinate dest, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;

    in_use[iSyn] = true;
    summationPoint[iSyn] = sum_point;
    summationCoord[iSyn] = dest;
    synapseCoord[iSyn] = source;
    W[iSyn] = 10.0e-9;
    this->type[iSyn] = type;
    U[iSyn] = DEFAULT_U;
    tau[iSyn] = DEFAULT_tau;

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

    this->U[iSyn] = U;
    this->D[iSyn] = D;
    this->F[iSyn] = F;

    this->tau[iSyn] = tau;
    total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    // initializes the queues for the Synapses
    initSpikeQueue(iSyn);
    // reset time varying state vars and recompute decay
    resetSynapse(iSyn, deltaT);
}

/**
 *  Advance one specific Synapse.
 *  @param  iSyn   index of the Synapse to connect to.
 *  @param  deltaT   inner simulation step duration
 */
void AllDSSynapses::advanceSynapse(const uint32_t iSyn, const BGFLOAT deltaT)
{
    uint64_t &lastSpike = this->lastSpike[iSyn];
    BGFLOAT &r = this->r[iSyn];
    BGFLOAT &u = this->u[iSyn];
    BGFLOAT &D = this->D[iSyn];
    BGFLOAT &F = this->F[iSyn];
    BGFLOAT &U = this->U[iSyn];
    BGFLOAT &W = this->W[iSyn];
    BGFLOAT &decay = this->decay[iSyn];
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &summationPoint = *(this->summationPoint[iSyn]);

    // is an input in the queue?
    if (isSpikeQueue(iSyn)) {
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
#endif // !defined(USE_GPU)
