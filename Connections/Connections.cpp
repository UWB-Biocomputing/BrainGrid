#include "Connections.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"

/* ------------- CONNECTIONS STRUCT ------------ *\
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
\* --------------------------------------------- */
/* ------------------- ERROR ------------------- *\
 * terminate called after throwing an instance of 'std::bad_alloc'
 *      what():  St9bad_alloc
 * ------------------- CAUSE ------------------- *|
 * As simulations expand in size the number of
 * neurons in total increases exponentially. When
 * using a MATRIX_TYPE = “complete” the amount of
 * used memory increases by another order of magnitude.
 * Once enough memory is used no more memory can be
 * allocated and a “bsd_alloc” will be thrown.
 * The following members of the connection constructor
 * consume equally vast amounts of memory as the
 * simulation sizes grow:
 *      - W             - radii
 *      - rates         - dist2
 *      - delta         - dist
 *      - areai
 * ----------------- 1/25/14 ------------------- *|
 * Currently when running a simulation of sizes
 * equal to or greater than 100 * 100 the above
 * error is thrown. After some testing we have
 * determined that this is a hardware dependent
 * issue, not software. We are also looking into
 * switching matrix types from "complete" to
 * "sparce". If successful it is possible the
 * problematic matricies mentioned above will use
 * only 1/250 of their current space.
\* --------------------------------------------- */
Connections::Connections() : nParams(0)
{
}

Connections::~Connections()
{
}

/*
 *  Update the connections status in every epoch.
 *
 *  @param  neurons  The Neuron list to search from.
 *  @param  sim_info SimulationInfo class to read information from.
 *  @param  layout   Layout information of the neunal network.
 *  @return true if successful, false otherwise.
 */
bool Connections::updateConnections(IAllNeurons &neurons, const SimulationInfo *sim_info, Layout *layout)
{
    return false;
}

#if defined(USE_GPU)
void Connections::updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeuronsDeviceProperties* m_allNeuronsDevice, AllSpikingSynapsesDeviceProperties* m_allSynapsesDevice, Layout *layout)
{
}
#else
/*
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *
 *  @param  num_neurons Number of neurons to update.
 *  @param  neurons     The Neuron list to search from.
 *  @param  synapses    The Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void Connections::updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, Layout *layout)
{
}
#endif // !USE_GPU


void Connections::createSynapsesFromWeights(const int num_neurons, const SimulationInfo *sim_info, Layout *layout, IAllNeurons &ineurons, IAllSynapses &isynapses) 
{
    AllNeurons &neurons = dynamic_cast<AllNeurons&>(ineurons);
    AllSynapses &synapses = dynamic_cast<AllSynapses&>(isynapses);
    
    // for each neuron
    for (int iNeuron = 0; iNeuron < num_neurons; iNeuron++) {
        // for each synapse in the neuron
        for (BGSIZE synapse_index = 0; synapse_index < sim_info->maxSynapsesPerNeuron; synapse_index++) {
            BGSIZE iSyn = sim_info->maxSynapsesPerNeuron * iNeuron + synapse_index;
            // if the synapse weight is not zero (which means there is a connection), create the synapse
            if(synapses.W[iSyn] != 0.0) {
                BGFLOAT theW = synapses.W[iSyn];
                int src_neuron = synapses.sourceNeuronIndex[iSyn];
                int dest_neuron = synapses.destNeuronIndex[iSyn];
                BGFLOAT* sum_point = &( neurons.summation_map[dest_neuron] );
                synapseType type = layout->synType(src_neuron, dest_neuron);
                synapses.synapse_counts[dest_neuron]++;
                synapses.createSynapse(iSyn, src_neuron, dest_neuron, sum_point, sim_info->deltaT, type);
                synapses.W[iSyn] = theW;
            }
        }
    }

}
