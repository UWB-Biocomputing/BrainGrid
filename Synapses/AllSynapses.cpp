#include "AllSynapses.h"
#include "AllNeurons.h"
#include "SynapseIndexMap.h"

// Default constructor
CUDA_CALLABLE AllSynapses::AllSynapses()
{
}

CUDA_CALLABLE AllSynapses::~AllSynapses()
{
}

#if defined(BOOST_PYTHON)
/*
 *  This function is called when Python variable that stores
 *  the synapses class object is destroyed.
 */
void AllSynapses::destroy()
{
}
#endif // BOOST_PYTHON

/*
 *  Assignment operator: copy synapses parameters.
 *
 *  @param  r_synapses  Synapses class object to copy from.
 */
IAllSynapses &AllSynapses::operator=(const IAllSynapses &r_synapses)
{
    m_pSynapsesProps->copyParameters(dynamic_cast<const AllSynapses &>(r_synapses).m_pSynapsesProps);

    return (*this);
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
    // allocate synspses properties data
    m_pSynapsesProps->setupSynapsesProps(num_neurons, max_synapses, sim_info, clr_info);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSynapses::cleanupSynapses()
{
    // deallocate neurons properties data
    delete m_pSynapsesProps;
    m_pSynapsesProps = NULL;
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllSynapses::checkNumParameters()
{
    return (m_pSynapsesProps->checkNumParameters());
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllSynapses::readParameters(const TiXmlElement& element)
{
    return (m_pSynapsesProps->readParameters(element));
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllSynapses::printParameters(ostream &output) const
{
    m_pSynapsesProps->printParameters(output);
}

/*
 *  Sets the data for Synapses to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSynapses::deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info)
{
        AllNeuronsProps *pNeuronsProps = dynamic_cast<AllNeurons&>(neurons).m_pNeuronsProps;

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
                BGSIZE iSyn = m_pSynapsesProps->maxSynapsesPerNeuron * neuron_index + synapse_index;

                m_pSynapsesProps->sourceNeuronLayoutIndex[iSyn] = neuron_index;

                m_pSynapsesProps->readSynapseProps(input, iSyn);

                m_pSynapsesProps->summationPoint[iSyn] = &(pNeuronsProps->summation_map[m_pSynapsesProps->destNeuronLayoutIndex[iSyn]]);

                read_synapses_counts[neuron_index]++;
        }

        for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
                m_pSynapsesProps->synapse_counts[i] = read_synapses_counts[i];
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
        synapse_count += m_pSynapsesProps->synapse_counts[i];
    }
    output << synapse_count << ends;

    for (int neuron_index = 0; neuron_index < clr_info->totalClusterNeurons; neuron_index++) {
        for (BGSIZE synapse_index = 0; synapse_index < m_pSynapsesProps->synapse_counts[neuron_index]; synapse_index++) {
            BGSIZE iSyn = m_pSynapsesProps->maxSynapsesPerNeuron * neuron_index + synapse_index;
            m_pSynapsesProps->writeSynapseProps(output, iSyn);
        }
    }
}

#if defined(USE_GPU)

/*
 *  Advance all the Synapses in the simulation.
 *  Update the state of all synapses for a time step.
 *
 *  @param  allNeuronsProps        Reference to the allNeurons struct on device memory.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  sim_info               SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 *  @param  synapsesDevice         Pointer to the Synapses object in device memory.
 *  @param  neuronsDevice          Pointer to the Neurons object in device memory.
 */
void AllSynapses::advanceSynapses(void* allNeuronsProps, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset, IAllSynapses* synapsesDevice, IAllNeurons* neuronsDevice)
{
    BGSIZE total_synapse_counts = m_pSynapsesProps->total_synapse_counts;
    if (total_synapse_counts == 0)
        return;

    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));
    uint64_t simulationStep = g_simulationStep + iStepOffset;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, simulationStep, maxSpikes, sim_info->deltaT, iStepOffset, synapsesDevice, neuronsDevice, (IAllNeuronsProps*)allNeuronsProps );
}

#else // USE_GPU

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
    int maxSpikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    AllSynapsesProps* pSynapsesProps = m_pSynapsesProps;
    uint64_t simulationStep = g_simulationStep + iStepOffset;
    IAllNeuronsProps *pINeuronsProps = dynamic_cast<AllNeurons*>(neurons)->m_pNeuronsProps;

    for (BGSIZE i = 0; i < m_pSynapsesProps->total_synapse_counts; i++) {
        // advance one specific Synapse
        BGSIZE iSyn = synapseIndexMap->incomingSynapseIndexMap[i];
        advanceSynapse(iSyn, sim_info->deltaT, neurons, simulationStep, iStepOffset, maxSpikes, pINeuronsProps);

        // and apply the post spike response to the summation point
        BGFLOAT &summationPoint = *(pSynapsesProps->summationPoint[iSyn]);
        BGFLOAT &psr = pSynapsesProps->psr[iSyn];
        summationPoint += psr;
    }
}

#endif // !USE_GPU

/*
 *  Remove a synapse from the network.
 *
 *  @param  neuron_index   Index of a neuron to remove from.
 *  @param  iSyn           Index of a synapse to remove.
 */
CUDA_CALLABLE void AllSynapses::eraseSynapse(const int neuron_index, const BGSIZE iSyn)
{
    m_pSynapsesProps->synapse_counts[neuron_index]--;
    m_pSynapsesProps->in_use[iSyn] = false;
    m_pSynapsesProps->summationPoint[iSyn] = NULL;
}

/*
 *  Adds a Synapse to the model, connecting two Neurons.
 *
 *  @param  iSyn        Index of the synapse to be added.
 *  @param  type        The type of the Synapse to add.
 *  @param  src_neuron  The Neuron that sends to this Synapse (layout index).
 *  @param  dest_neuron The Neuron that receives from the Synapse (layout index).
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration
 *  @param  iNeuron     Index of the destination neuron in the cluster.
 */
CUDA_CALLABLE void AllSynapses::addSynapse(BGSIZE &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point, const BGFLOAT deltaT, int iNeuron)
{
    if (m_pSynapsesProps->synapse_counts[iNeuron] >= m_pSynapsesProps->maxSynapsesPerNeuron) {
        DEBUG ( printf("Neuron : %d ran out of space for new synapses.\n", dest_neuron); )
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapse_index;
    for (synapse_index = 0; synapse_index < m_pSynapsesProps->maxSynapsesPerNeuron; synapse_index++) {
        iSyn = m_pSynapsesProps->maxSynapsesPerNeuron * iNeuron + synapse_index;
        if (!m_pSynapsesProps->in_use[iSyn]) {
            break;
        }
    }

    m_pSynapsesProps->synapse_counts[iNeuron]++;

    // create a synapse
    createSynapse(iSyn, src_neuron, dest_neuron, sum_point, deltaT, type );
}

/*
 *  Get the sign of the synapseType.
 *
 *  @param    type    synapseType I to I, I to E, E to I, or E to E
 *  @return   1 or -1, or 0 if error
 */
CUDA_CALLABLE int AllSynapses::synSign(const synapseType type)
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

/**
 *  Returns the type of synapse at the given coordinates
 *
 *  @param    neuron_type_map  The neuron type map (INH, EXC).
 *  @param    src_neuron  integer that points to a Neuron in the type map as a source.
 *  @param    dest_neuron integer that points to a Neuron in the type map as a destination.
 *  @return type of the synapse.
 */
CUDA_CALLABLE synapseType AllSynapses::synType(neuronType* neuron_type_map, const int src_neuron, const int dest_neuron)
{
    if ( neuron_type_map[src_neuron] == INH && neuron_type_map[dest_neuron] == INH )
        return II;
    else if ( neuron_type_map[src_neuron] == INH && neuron_type_map[dest_neuron] == EXC )
        return IE;
    else if ( neuron_type_map[src_neuron] == EXC && neuron_type_map[dest_neuron] == INH )
        return EI;
    else if ( neuron_type_map[src_neuron] == EXC && neuron_type_map[dest_neuron] == EXC )
        return EE;

    return STYPE_UNDEF;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
CUDA_CALLABLE void AllSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    reinterpret_cast<AllSynapsesProps*>(m_pSynapsesProps)->psr[iSyn] = 0.0;
}

#if defined(USE_GPU)

/*
 * Delete an AllSynapses class object in device
 *
 * @param pAllSynapses_d    Pointer to the AllSynapses object to be deleted in device.
 */
void AllSynapses::deleteAllSynapsesInDevice(IAllSynapses* pAllSynapses_d)
{
    // delete AllSynapses object in device memory.
    deleteAllSynapsesDevice <<< 1, 1 >>> ( pAllSynapses_d );
}

/*
 *  Set neurons properties.
 *
 *  @param  pAllSynapsesProps  Pointer to the neurons properties.
 */
CUDA_CALLABLE void AllSynapses::setSynapsesProps(void *pAllSynapsesProps)
{
    m_pSynapsesProps = static_cast<AllSynapsesProps*>(pAllSynapsesProps);
}

__global__ void deleteAllSynapsesDevice(IAllSynapses *pAllSynapses)
{
    delete pAllSynapses;
}

/* --------------------------------------*\
|* # Global Functions for advanceSynapses
\* --------------------------------------*/

/*
 *  CUDA code for advancing spiking synapses.
 *  Perform updating synapses for one time step.
 *
 *  @param[in] total_synapse_counts  Number of synapses.
 *  @param[in] synapseIndexMapDevice Reference to the SynapseIndexMap on device memory.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 *  @param[in] synapsesDevice        Pointer to the Synapses object in device memory.
 *  @param[in] neuronsDevice         Pointer to the Neurons object in device memory.
 *  @param[in] pINeuronsProps        Pointer to the neurons properties.
 */
__global__ void advanceSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, int maxSpikes, const BGFLOAT deltaT, int iStepOffset, IAllSynapses* synapsesDevice, IAllNeurons* neuronsDevice, IAllNeuronsProps* pINeuronsProps ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= total_synapse_counts )
                return;

        BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap[idx];

        synapsesDevice->advanceSynapse( iSyn, deltaT, neuronsDevice, simulationStep, iStepOffset, maxSpikes, pINeuronsProps );
}

#endif // USE_GPU
