#include "AllDSSynapses.h"

const BGFLOAT AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

AllDSSynapses::AllDSSynapses() : AllSynapses()
{
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
}

AllDSSynapses::AllDSSynapses(const int num_neurons, const int max_synapses) :
        AllSynapses(num_neurons, max_synapses)
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
    AllSynapses::setupSynapses(num_neurons, max_synapses);

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

    AllSynapses::cleanupSynapses();
}

void AllDSSynapses::readSynapses(istream& input, AllNeurons &neurons, const SimulationInfo *sim_info)
{
        // read the synapse data & create synapses
        int* read_synapses_counts= new int[sim_info->totalNeurons];
        for (int i = 0; i < sim_info->totalNeurons; i++) {
                read_synapses_counts[i] = 0;
        }

        int synapse_count;
        input >> synapse_count; input.ignore();
        for (int i = 0; i < synapse_count; i++) {
                // read the synapse data and add it to the list
                // create synapse
                Coordinate synapseCoord_coord;
                input >> synapseCoord_coord.x; input.ignore();
                input >> synapseCoord_coord.y; input.ignore();

                int neuron_index = synapseCoord_coord.x + synapseCoord_coord.y * sim_info->width;
                int synapse_index = read_synapses_counts[neuron_index];
                uint32_t iSyn = maxSynapsesPerNeuron * neuron_index + synapse_index;

                synapseCoord[iSyn] = synapseCoord_coord;

                readSynapse(input, iSyn, sim_info->deltaT);

                summationPoint[iSyn] =
                                &(neurons.summation_map[summationCoord[iSyn].x
                                + summationCoord[iSyn].y * sim_info->width]);

                read_synapses_counts[neuron_index]++;
        }

        for (int i = 0; i < sim_info->totalNeurons; i++) {
                        synapse_counts[i] = read_synapses_counts[i];
        }
        delete[] read_synapses_counts;
}

/*
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration.
 */
void AllDSSynapses::readSynapse(istream &input, const uint32_t iSyn, const BGFLOAT deltaT)
{
    int synapse_type(0);

    // input.ignore() so input skips over end-of-line characters.
    input >> summationCoord[iSyn].x; input.ignore();
    input >> summationCoord[iSyn].y; input.ignore();
    input >> W[iSyn]; input.ignore();
    input >> psr[iSyn]; input.ignore();
    input >> decay[iSyn]; input.ignore();
    input >> total_delay[iSyn]; input.ignore();
    input >> delayQueue[iSyn]; input.ignore();
    input >> delayIdx[iSyn]; input.ignore();
    input >> ldelayQueue[iSyn]; input.ignore();
    input >> synapse_type; input.ignore();
    input >> tau[iSyn]; input.ignore();
    input >> r[iSyn]; input.ignore();
    input >> u[iSyn]; input.ignore();
    input >> D[iSyn]; input.ignore();
    input >> U[iSyn]; input.ignore();
    input >> F[iSyn]; input.ignore();
    input >> lastSpike[iSyn]; input.ignore();
    input >> in_use[iSyn]; input.ignore();

    type[iSyn] = synapseOrdinalToType(synapse_type);
}

void AllDSSynapses::writeSynapses(ostream& output, const SimulationInfo *sim_info)
{
    // write the synapse data
    int synapse_count = 0;
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        synapse_count += synapse_counts[i];
    }
    output << synapse_count << ends;

    for (int neuron_index = 0; neuron_index < sim_info->totalNeurons; neuron_index++) {
        for (size_t synapse_index = 0; synapse_index < synapse_counts[neuron_index]; synapse_index++) {
            uint32_t iSyn = maxSynapsesPerNeuron * neuron_index + synapse_index;
            writeSynapse(output, iSyn);
        }
    }
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  iSyn   index of the synapse to print out.
 */
void AllDSSynapses::writeSynapse(ostream& output, const uint32_t iSyn) const 
{
    output << synapseCoord[iSyn].x << ends;
    output << synapseCoord[iSyn].y << ends;
    output << summationCoord[iSyn].x << ends;
    output << summationCoord[iSyn].y << ends;
    output << W[iSyn] << ends;
    output << psr[iSyn] << ends;
    output << decay[iSyn] << ends;
    output << total_delay[iSyn] << ends;
    output << delayQueue[iSyn] << ends;
    output << delayIdx[iSyn] << ends;
    output << ldelayQueue[iSyn] << ends;
    output << type[iSyn] << ends;
    output << tau[iSyn] << ends;
    output << r[iSyn] << ends;
    output << u[iSyn] << ends;
    output << D[iSyn] << ends;
    output << U[iSyn] << ends;
    output << F[iSyn] << ends;
    output << lastSpike[iSyn] << ends;
    output << in_use[iSyn] << ends;
}

/**     
 *  Returns an appropriate synapseType object for the given integer.
 *  @param  type_ordinal    integer that correspond with a synapseType.
 *  @return the SynapseType that corresponds with the given integer.
 */             
synapseType AllDSSynapses::synapseOrdinalToType(const int type_ordinal)
{
        switch (type_ordinal) {
        case 0:
                return II;
        case 1:
                return IE;
        case 2: 
                return EI; 
        case 3: 
                return EE;
        default:
                return STYPE_UNDEF;
        }       
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

/**
 *  Updates the decay if the synapse selected.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT  inner simulation step duration
 */
bool AllDSSynapses::updateDecay(const uint32_t iSyn, const BGFLOAT deltaT)
{
        BGFLOAT &tau = this->tau[iSyn];
        BGFLOAT &decay = this->decay[iSyn];

        if (tau > 0) {
                decay = exp( -deltaT / tau );
                return true;
        }
        return false;
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

