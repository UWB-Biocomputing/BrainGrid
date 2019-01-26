#include "AllDynamicSTDPSynapsesProps.h"

// Default constructor
AllDynamicSTDPSynapsesProps::AllDynamicSTDPSynapsesProps()
{
    lastSpike = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
}

AllDynamicSTDPSynapsesProps::~AllDynamicSTDPSynapsesProps()
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
void AllDynamicSTDPSynapsesProps::setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSTDPSynapsesProps::setupSynapsesProps(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        lastSpike = new uint64_t[max_total_synapses];
        r = new BGFLOAT[max_total_synapses];
        u = new BGFLOAT[max_total_synapses];
        D = new BGFLOAT[max_total_synapses];
        U = new BGFLOAT[max_total_synapses];
        F = new BGFLOAT[max_total_synapses];
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllDynamicSTDPSynapsesProps::cleanupSynapsesProps()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

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
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllDynamicSTDPSynapsesProps::readSynapseProps(istream &input, const BGSIZE iSyn)
{
    AllSTDPSynapsesProps::readSynapseProps(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> lastSpike[iSyn]; input.ignore();
    input >> r[iSyn]; input.ignore();
    input >> u[iSyn]; input.ignore();
    input >> D[iSyn]; input.ignore();
    input >> U[iSyn]; input.ignore();
    input >> F[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllDynamicSTDPSynapsesProps::writeSynapseProps(ostream& output, const BGSIZE iSyn) const
{
    AllSTDPSynapsesProps::writeSynapseProps(output, iSyn);

    output << lastSpike[iSyn] << ends;
    output << r[iSyn] << ends;
    output << u[iSyn] << ends;
    output << D[iSyn] << ends;
    output << U[iSyn] << ends;
    output << F[iSyn] << ends;
}
