/*
 *      \file SInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "SInputPoisson.h"
#include "tinyxml.h"
#include "AllDSSynapses.h"

extern void getValueList(const string& valString, vector<BGFLOAT>* pList);

/*
 * constructor
 * @param[in] parms     Pointer to xml parms element
 */
SInputPoisson::SInputPoisson(SimulationInfo* psi, TiXmlElement* parms) :
    m_nISIs(NULL),
    m_masks(NULL)
{
    m_fSInput = false;

    // read fr_mean and weight
    TiXmlElement* temp = NULL;
    string sync;
    BGFLOAT fr_mean;	// firing rate (per sec)

    if (( temp = parms->FirstChildElement( "IntParams" ) ) != NULL) 
    { 
        if (temp->QueryFLOATAttribute("fr_mean", &fr_mean ) != TIXML_SUCCESS) {
            cerr << "error IntParams:fr_mean" << endl;
            return;
        }
        if (temp->QueryFLOATAttribute("weight", &m_weight ) != TIXML_SUCCESS) {
            cerr << "error IntParams:weight" << endl;
            return;
        }
    }
    else
    {
        cerr << "missing IntParams" << endl;
        return;
    }

     // initialize firng rate, inverse firing rate
    fr_mean = fr_mean / 1000;	// firing rate per msec
    m_lambda = 1 / fr_mean;	// inverse firing rate

    // allocate memory for interval counter
    m_nISIs = new int[psi->totalNeurons];
    memset(m_nISIs, 0, sizeof(int) * psi->totalNeurons);
    
    // allocate memory for input masks
    m_masks = new bool[psi->totalNeurons];

    // read mask values and set it to masks
    vector<BGFLOAT> maskIndex;
    if ((temp = parms->FirstChildElement( "Masks")) != NULL)
    {
       TiXmlNode* pNode = NULL;
        while ((pNode = temp->IterateChildren(pNode)) != NULL)
        {
            if (strcmp(pNode->Value(), "M") == 0)
            {
                getValueList(pNode->ToElement()->GetText(), &maskIndex);

                memset(m_masks, false, sizeof(bool) * psi->totalNeurons);
                for (uint32_t i = 0; i < maskIndex.size(); i++)
                    m_masks[static_cast<int> ( maskIndex[i] )] = true;
            }
            else if (strcmp(pNode->Value(), "LayoutFiles") == 0)
            {
                string maskNListFileName;

                if (pNode->ToElement()->QueryValueAttribute( "maskNListFileName", &maskNListFileName ) == TIXML_SUCCESS)
                {
                    TiXmlDocument simDoc( maskNListFileName.c_str( ) );
                    if (!simDoc.LoadFile( ))
                    {
                        cerr << "Failed loading positions of stimulus input mask neurons list file " << maskNListFileName << ":" << "\n\t"
                             << simDoc.ErrorDesc( ) << endl;
                        cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                        break;
                    }
                    TiXmlNode* temp2 = NULL;
                    if (( temp2 = simDoc.FirstChildElement( "M" ) ) == NULL)
                    {
                        cerr << "Could not find <M> in positons of stimulus input mask neurons list file " << maskNListFileName << endl;
                        break;
                    }
                    getValueList(temp2->ToElement()->GetText(), &maskIndex);

                    memset(m_masks, false, sizeof(bool) * psi->totalNeurons);
                    for (uint32_t i = 0; i < maskIndex.size(); i++)
                        m_masks[static_cast<int> ( maskIndex[i] )] = true;
                }
            }
        }
    }
    else
    {
        // when no mask is specified, set it all true
        memset(m_masks, true, sizeof(bool) * psi->totalNeurons);
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
            AllSynapsesProperties *pSynapsesProperties = dynamic_cast<AllSynapsesProperties*>(pSynapses->m_pSynapsesProperties);
            pSynapsesProperties->W[iSyn] = m_weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
        }
    }
}

/*
 * Terminate process.
 *
 *  @param[in] psi             Pointer to the simulation information.
 *  @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void SInputPoisson::term(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
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
