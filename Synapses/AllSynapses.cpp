#include "AllSynapses.h"
#include "AllNeurons.h"

AllSynapses::AllSynapses() :
        total_synapse_counts(0),
        maxSynapsesPerNeuron(0),
        count_neurons(0),
        nParams(0)
{
    destNeuronIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronIndex = NULL;
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

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    BGSIZE max_total_synapses = max_synapses * num_neurons;

    maxSynapsesPerNeuron = max_synapses;
    total_synapse_counts = 0;
    count_neurons = num_neurons;

    if (max_total_synapses != 0) {
        destNeuronIndex = new int[max_total_synapses];
        W = new BGFLOAT[max_total_synapses];
        summationPoint = new BGFLOAT*[max_total_synapses];
        sourceNeuronIndex = new int[max_total_synapses];
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
        delete[] destNeuronIndex;
        delete[] W;
        delete[] summationPoint;
        delete[] sourceNeuronIndex;
        delete[] psr;
        delete[] type;
        delete[] in_use;
        delete[] synapse_counts;
    }

    destNeuronIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronIndex = NULL;
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
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSynapses::deserialize(istream& input, IAllNeurons &neurons, const SimulationInfo *sim_info)
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
                int neuron_index;
                input >> neuron_index; input.ignore();

                int synapse_index = read_synapses_counts[neuron_index];
                BGSIZE iSyn = maxSynapsesPerNeuron * neuron_index + synapse_index;

                sourceNeuronIndex[iSyn] = neuron_index;

                readSynapse(input, iSyn);

                summationPoint[iSyn] = &(dynamic_cast<AllNeurons&>(neurons).summation_map[destNeuronIndex[iSyn]]);

                read_synapses_counts[neuron_index]++;
        }

        for (int i = 0; i < sim_info->totalNeurons; i++) {
                        synapse_counts[i] = read_synapses_counts[i];
        }
        delete[] read_synapses_counts;
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSynapses::serialize(ostream& output, const SimulationInfo *sim_info)
{
    // write the synapse data
    int synapse_count = 0;
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        synapse_count += synapse_counts[i];
    }
    output << synapse_count << ends;

    for (int neuron_index = 0; neuron_index < sim_info->totalNeurons; neuron_index++) {
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
    input >> sourceNeuronIndex[iSyn]; input.ignore();
    input >> destNeuronIndex[iSyn]; input.ignore();
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
    output << sourceNeuronIndex[iSyn] << ends;
    output << destNeuronIndex[iSyn] << ends;
    output << W[iSyn] << ends;
    output << psr[iSyn] << ends;
    output << type[iSyn] << ends;
    output << in_use[iSyn] << ends;
}

/*
 *  Create a synapse index map.
 *
 *  @param  synapseIndexMap   Reference to the pointer to SynapseIndexMap structure.
 *  @param  sim_info          Pointer to the simulation information.
 */
void AllSynapses::createSynapseImap(SynapseIndexMap *&synapseIndexMap, const SimulationInfo* sim_info)
{
        int neuron_count = sim_info->totalNeurons;
        int total_synapse_counts = 0;

        // count the total synapses
        for ( int i = 0; i < neuron_count; i++ )
        {
                assert( static_cast<int>(synapse_counts[i]) < sim_info->maxSynapsesPerNeuron );
                total_synapse_counts += synapse_counts[i];
        }

        DEBUG ( cout << "total_synapse_counts: " << total_synapse_counts << endl; )

        if ( total_synapse_counts == 0 )
        {
                return;
        }

        // allocate memories for forward map
        vector<BGSIZE>* rgSynapseSynapseIndexMap = new vector<BGSIZE>[neuron_count];

        if (synapseIndexMap != NULL)
        {
            delete synapseIndexMap;
            synapseIndexMap = NULL;
        }

        BGSIZE syn_i = 0;
        int n_inUse = 0;

        // create synapse forward map & active synapse map
        synapseIndexMap = new SynapseIndexMap(neuron_count, total_synapse_counts);
        for (int i = 0; i < neuron_count; i++)
        {
                BGSIZE synapse_count = 0;
                synapseIndexMap->incomingSynapseBegin[i] = n_inUse;
                for ( int j = 0; j < sim_info->maxSynapsesPerNeuron; j++, syn_i++ )
                {
                        if ( in_use[syn_i] == true )
                        {
                                int idx = sourceNeuronIndex[syn_i];
                                rgSynapseSynapseIndexMap[idx].push_back(syn_i);

                                synapseIndexMap->incomingSynapseIndexMap[n_inUse] = syn_i;
                                n_inUse++;
                                synapse_count++;
                        }
                }
                assert( synapse_count == this->synapse_counts[i] );
                synapseIndexMap->incomingSynapseCount[i] = synapse_count;
        }

        assert( total_synapse_counts == n_inUse );
        this->total_synapse_counts = total_synapse_counts;

        syn_i = 0;
        for (int i = 0; i < neuron_count; i++)
        {
                synapseIndexMap->outgoingSynapseBegin[i] = syn_i;
                synapseIndexMap->outgoingSynapseCount[i] = rgSynapseSynapseIndexMap[i].size();

                for ( BGSIZE j = 0; j < rgSynapseSynapseIndexMap[i].size(); j++, syn_i++)
                {
                        synapseIndexMap->outgoingSynapseIndexMap[syn_i] = rgSynapseSynapseIndexMap[i][j];
                }
        }

        // delete memories
        delete[] rgSynapseSynapseIndexMap;
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
 *  @param  sim_info          SimulationInfo class to read information from.
 *  @param  neurons           The Neuron list to search from.
 *  @param  synapseIndexMap   Pointer to SynapseIndexMap structure.
 */
void AllSynapses::advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap)
{
    for (BGSIZE i = 0; i < total_synapse_counts; i++) {
        BGSIZE iSyn = synapseIndexMap->incomingSynapseIndexMap[i];
        advanceSynapse(iSyn, sim_info, neurons);
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
 */
void AllSynapses::addSynapse(BGSIZE &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point, const BGFLOAT deltaT)
{
    if (synapse_counts[dest_neuron] >= maxSynapsesPerNeuron) {
        DEBUG ( cout << "Neuron : " << dest_neuron << " ran out of space for new synapses." << endl; )
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapse_index;
    for (synapse_index = 0; synapse_index < maxSynapsesPerNeuron; synapse_index++) {
        iSyn = maxSynapsesPerNeuron * dest_neuron + synapse_index;
        if (!in_use[iSyn]) {
            break;
        }
    }

    synapse_counts[dest_neuron]++;

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

