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
 *  Creates synapses from synapse weights saved in the serialization file.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void Connections::createSynapsesFromWeights(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo) 
{
    // for each each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        AllNeurons *neurons = dynamic_cast<AllNeurons*>(vtClr[iCluster]->m_neurons);
        AllNeuronsProps *pNeuronsProps = neurons->m_pNeuronsProps;
        AllSynapses *synapses = dynamic_cast<AllSynapses*>(vtClr[iCluster]->m_synapses);
        AllSynapsesProps *pSynapsesProps = synapses->m_pSynapsesProps;

        // for each neuron in the cluster
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++) {
            // for each synapse in the neuron
            for (BGSIZE synapse_index = 0; synapse_index < pSynapsesProps->maxSynapsesPerNeuron; synapse_index++) {
                BGSIZE iSyn = pSynapsesProps->maxSynapsesPerNeuron * iNeuron + synapse_index;
                // if the synapse weight is not zero (which means there is a connection), create the synapse
                if(pSynapsesProps->W[iSyn] != 0.0) {
                    BGFLOAT theW = pSynapsesProps->W[iSyn];
                    BGFLOAT* sum_point = &( pNeuronsProps->summation_map[iNeuron] );
                    int src_neuron = pSynapsesProps->sourceNeuronLayoutIndex[iSyn];
                    int dest_neuron = pSynapsesProps->destNeuronLayoutIndex[iSyn];
                    synapseType type = synapses->synType(layout->neuron_type_map, src_neuron, dest_neuron);
                    pSynapsesProps->synapse_counts[iNeuron]++;
                    synapses->createSynapse(iSyn, src_neuron, dest_neuron, sum_point, sim_info->deltaT, type);
                    pSynapsesProps->W[iSyn] = theW;
                }
            }
        }
    }
}