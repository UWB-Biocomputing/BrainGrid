#include "AllSynapses.h"
#include "AllNeurons.h"

AllSynapses::AllSynapses() :
        maxSynapsesPerNeuron(0),
        total_synapse_counts(0),
        count_neurons(0)
{
    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;
}

AllSynapses::AllSynapses(const int num_neurons, const int max_synapses) 
{
    setupSynapses(num_neurons, max_synapses);
}

AllSynapses::~AllSynapses()
{
    cleanupSynapses();
}

void AllSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

void AllSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    uint32_t max_total_synapses = max_synapses * num_neurons;

    maxSynapsesPerNeuron = max_synapses;
    total_synapse_counts = 0;
    count_neurons = num_neurons;

    if (max_total_synapses != 0) {
        summationCoord = new Coordinate[max_total_synapses];
        W = new BGFLOAT[max_total_synapses];
        summationPoint = new BGFLOAT*[max_total_synapses];
        synapseCoord = new Coordinate[max_total_synapses];
        psr = new BGFLOAT[max_total_synapses];
        type = new synapseType[max_total_synapses];
        in_use = new bool[max_total_synapses];
        synapse_counts = new size_t[num_neurons];

        for (int i = 0; i < max_total_synapses; i++) {
            summationPoint[i] = NULL;
            in_use[i] = false;
        }

        for (int i = 0; i < num_neurons; i++) {
            synapse_counts[i] = 0;
        }
    }
}

void AllSynapses::cleanupSynapses()
{
    uint32_t max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] summationCoord;
        delete[] W;
        delete[] summationPoint;
        delete[] synapseCoord;
        delete[] psr;
        delete[] type;
        delete[] in_use;
        delete[] synapse_counts;
    }

    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;

    count_neurons = 0;
    maxSynapsesPerNeuron = 0;
}

/**
 *  Reset time varying state vars and recompute decay.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration
 */
void AllSynapses::resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT)
{
    psr[iSyn] = 0.0;
}

void AllSynapses::readSynapses(istream& input, AllNeurons &neurons, const SimulationInfo *sim_info)
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

                readSynapse(input, iSyn);

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

void AllSynapses::writeSynapses(ostream& output, const SimulationInfo *sim_info)
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

/*
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  iSyn   index of the synapse to set.
 */
void AllSynapses::readSynapse(istream &input, const uint32_t iSyn)
{
    int synapse_type(0);

    // input.ignore() so input skips over end-of-line characters.
    input >> summationCoord[iSyn].x; input.ignore();
    input >> summationCoord[iSyn].y; input.ignore();
    input >> W[iSyn]; input.ignore();
    input >> psr[iSyn]; input.ignore();
    input >> synapse_type; input.ignore();
    input >> in_use[iSyn]; input.ignore();

    type[iSyn] = synapseOrdinalToType(synapse_type);
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  iSyn   index of the synapse to print out.
 */
void AllSynapses::writeSynapse(ostream& output, const uint32_t iSyn) const
{
    output << synapseCoord[iSyn].x << ends;
    output << synapseCoord[iSyn].y << ends;
    output << summationCoord[iSyn].x << ends;
    output << summationCoord[iSyn].y << ends;
    output << W[iSyn] << ends;
    output << psr[iSyn] << ends;
    output << type[iSyn] << ends;
    output << in_use[iSyn] << ends;
}

/**     
 *  Returns an appropriate synapseType object for the given integer.
 *  @param  type_ordinal    integer that correspond with a synapseType.
 *  @return the SynapseType that corresponds with the given integer.
 */
synapseType AllSynapses::synapseOrdinalToType(const int type_ordinal)
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

#if !defined(USE_GPU)
/**
 *  Advance all the Synapses in the simulation.
 *  param  sim_info    SimulationInfo class to read information from.
 */
void AllSynapses::advanceSynapses(const SimulationInfo *sim_info, AllNeurons *neurons)
{
    int num_neurons = sim_info->totalNeurons;
    BGFLOAT deltaT = sim_info->deltaT;

    for (int i = 0; i < num_neurons; i++) {
        size_t synapse_counts = this->synapse_counts[i];
        int synapse_advanced = 0;
        for (int z = 0; z < synapse_counts; z++) {
            // Advance Synapse
            uint32_t iSyn = maxSynapsesPerNeuron * i + z;
            advanceSynapse(iSyn, sim_info, neurons);
            synapse_advanced++;
        }
    }
}

/**
 *  Remove a synapse from the network.
 *  @param  neuron_index   Index of a neuron.
 *  @param  iSyn      Index of a synapse.
 */
void AllSynapses::eraseSynapse(const int neuron_index, const uint32_t iSyn)
{
    synapse_counts[neuron_index]--;
    in_use[iSyn] = false;
    summationPoint[iSyn] = NULL;
}

/**
 *  Adds a Synapse to the model, connecting two Neurons.
 *  @param  type    the type of the Synapse to add.
 *  @param  type    the weight of the Synapse.
 *  @param  src_neuron  the Neuron that sends to this Synapse.
 *  @param  dest_neuron the Neuron that receives from the Synapse.
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param deltaT   inner simulation step duration
 */
void AllSynapses::addSynapse(BGFLOAT weight, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, const BGFLOAT deltaT)
{
    if (synapse_counts[src_neuron] >= maxSynapsesPerNeuron) {
        return; // TODO: ERROR!
    }

    // add it to the list
    size_t synapse_index;
    uint32_t iSyn;
    for (synapse_index = 0; synapse_index < maxSynapsesPerNeuron; synapse_index++) {
        iSyn = maxSynapsesPerNeuron * src_neuron + synapse_index;
        if (!in_use[iSyn]) {
            break;
        }
    }

    synapse_counts[src_neuron]++;

    // create a synapse
    createSynapse(iSyn, source, dest, sum_point, deltaT, type );

    W[iSyn] = weight;
}
#endif // !defined(USE_GPU)
