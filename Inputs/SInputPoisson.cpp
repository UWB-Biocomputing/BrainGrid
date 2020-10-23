/*
 *      \file SInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "SInputPoisson.h"
#include "AllDSSynapses.h"

/*
 * constructor
 * @param[in] psi       Pointer to the simulation information parms element
 * @param[in] fr_mean   Firing rate (per sec)
 * @param[in] weight    Synapse weight
 * @param[in] maskIndex Input masks index
 */
SInputPoisson::SInputPoisson(SimulationInfo* psi, BGFLOAT fr_mean, BGFLOAT weight, vector<BGFLOAT> &maskIndex) :
    m_nISIs(NULL),
    m_masks(NULL)
{
    m_fSInput = false;
    m_weight = weight;

     // initialize firng rate, inverse firing rate
    fr_mean = fr_mean / 1000;	// firing rate per msec
    m_lambda = 1 / fr_mean;	// inverse firing rate

    // allocate memory for interval counter
    m_nISIs = new int[psi->totalNeurons];
    memset(m_nISIs, 0, sizeof(int) * psi->totalNeurons);
    
    // allocate memory for input masks
    m_masks = new bool[psi->totalNeurons];

    // set mask values
    memset(m_masks, false, sizeof(bool) * psi->totalNeurons);
    if (maskIndex.size() == 0)
    { 
        // when no mask is specified, set it all true
        memset(m_masks, true, sizeof(bool) * psi->totalNeurons);
    }
    else
    {
        for (uint32_t i = 0; i < maskIndex.size(); i++)
            m_masks[static_cast<int> ( maskIndex[i] )] = true;
    }

    m_fSInput = true;
}

/*
 * destructor
 */
SInputPoisson::~SInputPoisson()
{
}

/*
 * Initialize data.
 *
 *  @param[in] psi            Pointer to the simulation information.
 *  @param[in] vtClrInfo      Vector of ClusterInfo.
 */
void SInputPoisson::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (m_fSInput == false)
        return;

    m_maxSynapsesPerNeuron = 1;

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++) 
    {
        ClusterInfo *clr_info = vtClrInfo[iCluster];
        
        int neuronLayoutIndex = clr_info->clusterNeuronsBegin;
        int totalClusterNeurons = clr_info->totalClusterNeurons;

        // create an input synapse layer
        // TODO: do we need to support other types of synapses?
        clr_info->synapsesSInput = new AllDSSynapses();
        clr_info->synapsesSInput->createSynapsesProps();

        // HACK!!! avoid to overwrite eventHandler in setupSynapses
        InterClustersEventHandler* t_eventHandler = clr_info->eventHandler;
        clr_info->eventHandler = NULL;
        clr_info->synapsesSInput->setupSynapses(totalClusterNeurons, m_maxSynapsesPerNeuron, psi, vtClrInfo[iCluster]);
        clr_info->eventHandler = t_eventHandler;

        // for each neuron in the cluster
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++)
        {
            synapseType type;
            if (psi->model->getLayout()->neuron_type_map[neuronLayoutIndex] == INH)
                type = EI;
            else
                type = EE;

            BGFLOAT* sum_point = &( clr_info->pClusterSummationMap[iNeuron] );
            BGSIZE iSyn = m_maxSynapsesPerNeuron * iNeuron;

            // create a Synapse and connect it to the model.
            // 0 - source neuron index, iNeuron - destination nenuron index.
            (clr_info->synapsesSInput)->createSynapse(iSyn, 0, iNeuron, sum_point, psi->deltaT, type);
            AllSynapses *pSynapses = dynamic_cast<AllSynapses*>(clr_info->synapsesSInput);
            AllSynapsesProps *pSynapsesProps = dynamic_cast<AllSynapsesProps*>(pSynapses->m_pSynapsesProps);
            pSynapsesProps->W[iSyn] = m_weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
        }
    }
}

/*
 * Terminate process.
 *
 *  @param[in] psi             Pointer to the simulation information.
 *  @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void SInputPoisson::term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo)
{
    // clear memory for interval counter
    if (m_nISIs != NULL)
        delete[] m_nISIs;

    // clear the synapse layer, which destroy all synase objects
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++)
    {
        if (vtClrInfo[iCluster]->synapsesSInput != NULL)
            delete vtClrInfo[iCluster]->synapsesSInput;
    }

    // clear memory for input masks
    if (m_masks != NULL)
        delete[] m_masks;
}
