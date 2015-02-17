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

    r = new BGFLOAT*[num_neurons];
    u = new BGFLOAT*[num_neurons];
    D = new BGFLOAT*[num_neurons];
    U = new BGFLOAT*[num_neurons];
    F = new BGFLOAT*[num_neurons];

    if (max_synapses != 0) {
        for (int i = 0; i < num_neurons; i++) {
            r[i] = new BGFLOAT[max_synapses];
            u[i] = new BGFLOAT[max_synapses];
            D[i] = new BGFLOAT[max_synapses];
            U[i] = new BGFLOAT[max_synapses];
            F[i] = new BGFLOAT[max_synapses];
        }
    }
}

void AllDSSynapses::cleanupSynapses()
{
    if (count_neurons != 0) {
        if (max_synapses != 0) {
            for (int i = 0; i < count_neurons; i++) {
                delete[] r[i];
                delete[] u[i];
                delete[] D[i];
                delete[] U[i];
                delete[] F[i];
            }
        }

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
                int synapses_index = read_synapses_counts[neuron_index];

                synapseCoord[neuron_index][synapses_index] = synapseCoord_coord;

                readSynapse(input, neuron_index, synapses_index, sim_info->deltaT);

                summationPoint[neuron_index][synapses_index] =
                                &(neurons.summation_map[summationCoord[neuron_index][synapses_index].x
                                + summationCoord[neuron_index][synapses_index].y * sim_info->width]);

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
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration.
 */
void AllDSSynapses::readSynapse(istream &input, const int neuron_index, const int synapse_index, const BGFLOAT deltaT)
{
    int synapse_type(0);

    // input.ignore() so input skips over end-of-line characters.
    input >> summationCoord[neuron_index][synapse_index].x; input.ignore();
    input >> summationCoord[neuron_index][synapse_index].y; input.ignore();
    input >> W[neuron_index][synapse_index]; input.ignore();
    input >> psr[neuron_index][synapse_index]; input.ignore();
    input >> decay[neuron_index][synapse_index]; input.ignore();
    input >> total_delay[neuron_index][synapse_index]; input.ignore();
    input >> delayQueue[neuron_index][synapse_index][0]; input.ignore();
    input >> delayIdx[neuron_index][synapse_index]; input.ignore();
    input >> ldelayQueue[neuron_index][synapse_index]; input.ignore();
    input >> synapse_type; input.ignore();
    input >> tau[neuron_index][synapse_index]; input.ignore();
    input >> r[neuron_index][synapse_index]; input.ignore();
    input >> u[neuron_index][synapse_index]; input.ignore();
    input >> D[neuron_index][synapse_index]; input.ignore();
    input >> U[neuron_index][synapse_index]; input.ignore();
    input >> F[neuron_index][synapse_index]; input.ignore();
    input >> lastSpike[neuron_index][synapse_index]; input.ignore();
    input >> in_use[neuron_index][synapse_index]; input.ignore();

    type[neuron_index][synapse_index] = synapseOrdinalToType(synapse_type);
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
            writeSynapse(output, neuron_index, synapse_index);
        }
    }
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to print out.
 */
void AllDSSynapses::writeSynapse(ostream& output, const int neuron_index, const int synapse_index) const 
{
    output << synapseCoord[neuron_index][synapse_index].x << ends;
    output << synapseCoord[neuron_index][synapse_index].y << ends;
    output << summationCoord[neuron_index][synapse_index].x << ends;
    output << summationCoord[neuron_index][synapse_index].y << ends;
    output << W[neuron_index][synapse_index] << ends;
    output << psr[neuron_index][synapse_index] << ends;
    output << decay[neuron_index][synapse_index] << ends;
    output << total_delay[neuron_index][synapse_index] << ends;
    output << delayQueue[neuron_index][synapse_index][0] << ends;
    output << delayIdx[neuron_index][synapse_index] << ends;
    output << ldelayQueue[neuron_index][synapse_index] << ends;
    output << type[neuron_index][synapse_index] << ends;
    output << tau[neuron_index][synapse_index] << ends;
    output << r[neuron_index][synapse_index] << ends;
    output << u[neuron_index][synapse_index] << ends;
    output << D[neuron_index][synapse_index] << ends;
    output << U[neuron_index][synapse_index] << ends;
    output << F[neuron_index][synapse_index] << ends;
    output << lastSpike[neuron_index][synapse_index] << ends;
    output << in_use[neuron_index][synapse_index] << ends;
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
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration
 */
void AllDSSynapses::resetSynapse(const int neuron_index, const int synapse_index, const BGFLOAT deltaT)
{
        psr[neuron_index][synapse_index] = 0.0;
        assert( updateDecay(neuron_index, synapse_index, deltaT) );
        u[neuron_index][synapse_index] = DEFAULT_U;
        r[neuron_index][synapse_index] = 1.0;
        lastSpike[neuron_index][synapse_index] = ULONG_MAX;
}

/**
 *  Updates the decay if the synapse selected.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 *  @param  deltaT  inner simulation step duration
 */
bool AllDSSynapses::updateDecay(const int neuron_index, const int synapse_index, const BGFLOAT deltaT)
{
        BGFLOAT &tau = this->tau[neuron_index][synapse_index];
        BGFLOAT &decay = this->decay[neuron_index][synapse_index];

        if (tau > 0) {
                decay = exp( -deltaT / tau );
                return true;
        }
        return false;
}

/**
 *  Create a Synapse and connect it to the model.
 *  @param  synapses    the Neuron list to reference.
 *  @param  neuron_index    TODO 
 *  @param  synapse_index   TODO
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param  deltaT  TODO
 *  @param  type    type of the Synapse to create.
 */
void AllDSSynapses::createSynapse(const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;

    in_use[neuron_index][synapse_index] = true;
    summationPoint[neuron_index][synapse_index] = sum_point;
    summationCoord[neuron_index][synapse_index] = dest;
    synapseCoord[neuron_index][synapse_index] = source;
    W[neuron_index][synapse_index] = 10.0e-9;
    this->type[neuron_index][synapse_index] = type;
    U[neuron_index][synapse_index] = DEFAULT_U;
    tau[neuron_index][synapse_index] = DEFAULT_tau;

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

    this->U[neuron_index][synapse_index] = U;
    this->D[neuron_index][synapse_index] = D;
    this->F[neuron_index][synapse_index] = F;

    this->tau[neuron_index][synapse_index] = tau;
    total_delay[neuron_index][synapse_index] = static_cast<int>( delay / deltaT ) + 1;

    // initializes the queues for the Synapses
    initSpikeQueue(neuron_index, synapse_index);
    // reset time varying state vars and recompute decay
    resetSynapse(neuron_index, synapse_index, deltaT);
}

