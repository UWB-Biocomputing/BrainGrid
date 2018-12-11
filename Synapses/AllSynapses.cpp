#include "AllSynapses.h"
#include "AllNeurons.h"
#include "SynapseIndexMap.h"

// Default constructor
AllSynapses::AllSynapses() :
        total_synapse_counts(0),
        maxSynapsesPerNeuron(0),
        count_neurons(0),
        nParams(0)
{
    destNeuronLayoutIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronLayoutIndex = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;
}

// Copy constructor
AllSynapses::AllSynapses(const AllSynapses &r_synapses) :
        total_synapse_counts(0),
        maxSynapsesPerNeuron(0),
        count_neurons(0),
        nParams(0)
{
    destNeuronLayoutIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronLayoutIndex = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;
}

AllSynapses::~AllSynapses()
{
    cleanupSynapses();
}

/*
 *  Assignment operator: copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
IAllSynapses &AllSynapses::operator=(const IAllSynapses &r_synapses)
{
    copyParameters(dynamic_cast<const AllSynapses &>(r_synapses));

    return (*this);
}

/*
 *  Copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
void AllSynapses::copyParameters(const AllSynapses &r_synapses)
{
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSynapses::setupSynapses(SimulationInfo *sim_info, ClusterInfo *clr_info)
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
void AllSynapses::setupSynapses(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    BGSIZE max_total_synapses = max_synapses * num_neurons;

    maxSynapsesPerNeuron = max_synapses;
    total_synapse_counts = 0;
    count_neurons = num_neurons;

    if (max_total_synapses != 0) {
        destNeuronLayoutIndex = new int[max_total_synapses];
        W = new BGFLOAT[max_total_synapses];
        summationPoint = new BGFLOAT*[max_total_synapses];
        sourceNeuronLayoutIndex = new int[max_total_synapses];
        psr = new BGFLOAT[max_total_synapses];
        type = new synapseType[max_total_synapses];
        in_use = new bool[max_total_synapses];
        synapse_counts = new BGSIZE[num_neurons];

        for (BGSIZE i = 0; i < max_total_synapses; i++) {
            summationPoint[i] = NULL;
            in_use[i] = false;
        }

        for (int i = 0; i < num_neurons; i++) {
            synapse_counts[i] = 0;
        }
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSynapses::cleanupSynapses()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] destNeuronLayoutIndex;
        delete[] W;
        delete[] summationPoint;
        delete[] sourceNeuronLayoutIndex;
        delete[] psr;
        delete[] type;
        delete[] in_use;
        delete[] synapse_counts;
    }

    destNeuronLayoutIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronLayoutIndex = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;

    count_neurons = 0;
    maxSynapsesPerNeuron = 0;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
void AllSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    psr[iSyn] = 0.0;
}

/*
 *  Sets the data for Synapses to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSynapses::deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info)
{
        // read the synapse data & create synapses
        int* read_synapses_counts= new int[clr_info->totalClusterNeurons];
        for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
                read_synapses_counts[i] = 0;
        }

        int synapse_count;
        input >> synapse_count; input.ignore();
        for (int i = 0; i < synapse_count; i++) {
                // read the synapse data and add it to the list
                // create synapse
                int neuron_index;
                input >> neuron_index; input.ignore();

                int synapse_index = read_synapses_counts[neuron_index];
                BGSIZE iSyn = maxSynapsesPerNeuron * neuron_index + synapse_index;

                sourceNeuronLayoutIndex[iSyn] = neuron_index;

                readSynapse(input, iSyn);

                AllNeuronsProperties *pNeuronsProperties = dynamic_cast<AllNeuronsProperties*>(dynamic_cast<AllNeurons&>(neurons).m_pNeuronsProperties);
                summationPoint[iSyn] = &(pNeuronsProperties->summation_map[destNeuronLayoutIndex[iSyn]]);

                read_synapses_counts[neuron_index]++;
        }

        for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
                        synapse_counts[i] = read_synapses_counts[i];
        }
        delete[] read_synapses_counts;
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSynapses::serialize(ostream& output, const ClusterInfo *clr_info)
{
    // write the synapse data
    int synapse_count = 0;
    for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
        synapse_count += synapse_counts[i];
    }
    output << synapse_count << ends;

    for (int neuron_index = 0; neuron_index < clr_info->totalClusterNeurons; neuron_index++) {
        for (BGSIZE synapse_index = 0; synapse_index < synapse_counts[neuron_index]; synapse_index++) {
            BGSIZE iSyn = maxSynapsesPerNeuron * neuron_index + synapse_index;
            writeSynapse(output, iSyn);
        }
    }
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSynapses::readSynapse(istream &input, const BGSIZE iSyn)
{
    int synapse_type(0);

    // input.ignore() so input skips over end-of-line characters.
    input >> sourceNeuronLayoutIndex[iSyn]; input.ignore();
    input >> destNeuronLayoutIndex[iSyn]; input.ignore();
    input >> W[iSyn]; input.ignore();
    input >> psr[iSyn]; input.ignore();
    input >> synapse_type; input.ignore();
    input >> in_use[iSyn]; input.ignore();

    type[iSyn] = synapseOrdinalToType(synapse_type);
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSynapses::writeSynapse(ostream& output, const BGSIZE iSyn) const
{
    output << sourceNeuronLayoutIndex[iSyn] << ends;
    output << destNeuronLayoutIndex[iSyn] << ends;
    output << W[iSyn] << ends;
    output << psr[iSyn] << ends;
    output << type[iSyn] << ends;
    output << in_use[iSyn] << ends;
}

/*     
 *  Returns an appropriate synapseType object for the given integer.
 *
 *  @param  type_ordinal    Integer that correspond with a synapseType.
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
/*
 *  Advance all the Synapses in the simulation.
 *
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  neurons          The Neuron list to search from.
 *  @param  synapseIndexMap  Pointer to the synapse index map.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSynapses::advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset)
{
    for (BGSIZE i = 0; i < total_synapse_counts; i++) {
        BGSIZE iSyn = synapseIndexMap->incomingSynapseIndexMap[i];
        advanceSynapse(iSyn, sim_info, neurons, iStepOffset);
    }
}

/*
 *  Remove a synapse from the network.
 *
 *  @param  neuron_index   Index of a neuron to remove from.
 *  @param  iSyn           Index of a synapse to remove.
 */
void AllSynapses::eraseSynapse(const int neuron_index, const BGSIZE iSyn)
{
    synapse_counts[neuron_index]--;
    in_use[iSyn] = false;
    summationPoint[iSyn] = NULL;
}
#endif // !defined(USE_GPU)

/*
 *  Adds a Synapse to the model, connecting two Neurons.
 *
 *  @param  iSyn        Index of the synapse to be added.
 *  @param  type        The type of the Synapse to add.
 *  @param  src_neuron  The Neuron that sends to this Synapse.
 *  @param  dest_neuron The Neuron that receives from the Synapse.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration
 *  @param  clr_info    ClusterInfo to refer from.
 */
void AllSynapses::addSynapse(BGSIZE &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point, const BGFLOAT deltaT, const ClusterInfo *clr_info)
{
    int iNeuron = dest_neuron - clr_info->clusterNeuronsBegin;
 
    if (synapse_counts[iNeuron] >= maxSynapsesPerNeuron) {
        DEBUG ( cout << "Neuron : " << dest_neuron << " ran out of space for new synapses." << endl; )
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapse_index;
    for (synapse_index = 0; synapse_index < maxSynapsesPerNeuron; synapse_index++) {
        iSyn = maxSynapsesPerNeuron * iNeuron + synapse_index;
        if (!in_use[iSyn]) {
            break;
        }
    }

    synapse_counts[iNeuron]++;

    // create a synapse
    createSynapse(iSyn, src_neuron, dest_neuron, sum_point, deltaT, type );
}

/*
 *  Get the sign of the synapseType.
 *
 *  @param    type    synapseType I to I, I to E, E to I, or E to E
 *  @return   1 or -1, or 0 if error
 */
int AllSynapses::synSign(const synapseType type)
{
    switch ( type ) {
        case II:
        case IE:
            return -1;
        case EI:
        case EE:
            return 1;
        case STYPE_UNDEF:
            // TODO error.
            return 0;
    }

    return 0;
}

