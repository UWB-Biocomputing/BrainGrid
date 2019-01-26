#include "AllSynapsesProps.h"

// Default constructor
AllSynapsesProps::AllSynapsesProps()
{
    nParams = 0;

    destNeuronLayoutIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronLayoutIndex = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;
}

AllSynapsesProps::~AllSynapsesProps()
{
    cleanupSynapsesProps();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSynapsesProps::setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    count_neurons = num_neurons;
    maxSynapsesPerNeuron = max_synapses;
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    total_synapse_counts = 0;

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
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSynapsesProps::cleanupSynapsesProps()
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
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllSynapsesProps::checkNumParameters()
{
    return (nParams >= 0);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllSynapsesProps::readParameters(const TiXmlElement& element)
{
    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllSynapsesProps::printParameters(ostream &output) const
{
}

/*
 *  Copy synapses parameters.
 *
 *  @param  r_synapsesProps  Synapses properties class object to copy from.
 */
void AllSynapsesProps::copyParameters(const AllSynapsesProps *r_synapsesProps)
{
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSynapsesProps::readSynapseProps(istream &input, const BGSIZE iSyn)
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
void AllSynapsesProps::writeSynapseProps(ostream& output, const BGSIZE iSyn) const
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
synapseType AllSynapsesProps::synapseOrdinalToType(const int type_ordinal)
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
