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
    nISIs(NULL),
    m_synapses(NULL),
    masks(NULL)
{
    fSInput = false;

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
        if (temp->QueryFLOATAttribute("weight", &weight ) != TIXML_SUCCESS) {
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
    lambda = 1 / fr_mean;	// inverse firing rate

    // allocate memory for interval counter
    nISIs = new int[psi->totalNeurons];
    memset(nISIs, 0, sizeof(int) * psi->totalNeurons);
    
    // allocate memory for input masks
    masks = new bool[psi->totalNeurons];

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

                memset(masks, false, sizeof(bool) * psi->totalNeurons);
                for (uint32_t i = 0; i < maskIndex.size(); i++)
                    masks[static_cast<int> ( maskIndex[i] )] = true;
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

                    memset(masks, false, sizeof(bool) * psi->totalNeurons);
                    for (uint32_t i = 0; i < maskIndex.size(); i++)
                        masks[static_cast<int> ( maskIndex[i] )] = true;
                }
            }
        }
    }
    else
    {
        // when no mask is specified, set it all true
        memset(masks, true, sizeof(bool) * psi->totalNeurons);
    }

    fSInput = true;
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
 *  @param[in] psi       Pointer to the simulation information.
 */
void SInputPoisson::init(SimulationInfo* psi)
{
    if (fSInput == false)
        return;

    // create an input synapse layer
    // TODO: do we need to support other types of synapses?
    m_synapses = new AllDSSynapses(psi->totalNeurons, 1);
    for (int neuron_index = 0; neuron_index < psi->totalNeurons; neuron_index++)
    {
        synapseType type;
        if (psi->model->getLayout()->neuron_type_map[neuron_index] == INH)
            type = EI;
        else
            type = EE;

        BGFLOAT* sum_point = &( psi->pSummationMap[neuron_index] );
        BGSIZE iSyn = psi->maxSynapsesPerNeuron * neuron_index;

        m_synapses->createSynapse(iSyn, 0, neuron_index, sum_point, psi->deltaT, type);
        dynamic_cast<AllSynapses*>(m_synapses)->W[iSyn] = weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
    }
}

/*
 * Terminate process.
 *
 *  @param[in] psi       Pointer to the simulation information.
 */
void SInputPoisson::term(SimulationInfo* psi)
{
    // clear memory for interval counter
    if (nISIs != NULL)
        delete[] nISIs;

    // clear the synapse layer, which destroy all synase objects
    if (m_synapses != NULL)
        delete m_synapses;

    // clear memory for input masks
    if (masks != NULL)
        delete[] masks;
}
