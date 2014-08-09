/**
 *      \file SInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "SInputPoisson.h"
#include "tinyxml/tinyxml.h"

/**
 * constructor
 */
SInputPoisson::SInputPoisson()
{
    
}

/**
 * destructor
 */
SInputPoisson::~SInputPoisson()
{
}

/**
 * Initialize data.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] parms     Pointer to xml parms element
 */
void SInputPoisson::init(SimulationInfo* psi, TiXmlElement* parms)
{
    fSInput = false;

    // read fr_mean and weight
    TiXmlElement* temp = NULL;
    string sync;
    BGFLOAT fr_mean;	// firing rate (per sec)
    BGFLOAT weight;	// synapse weight

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
    nISIs = new int[psi->cNeurons];
    memset(nISIs, 0, sizeof(int) * psi->cNeurons);
    
    // create an input synapse layer
    synapseList.clear();
    synapseList.reserve(psi->cNeurons);
    for (int i = 0; i < psi->cNeurons; i++)
    {
        // create a synapse
        int dest_x = i % psi->width;
        int dest_y = i / psi->width;
        BGFLOAT* sp = &(psi->pSummationMap[dest_x + dest_y * psi->width]);
        synapseType type = (psi->rgNeuronTypeMap[i] == INH ? EI : EE);
        DynamicSpikingSynapse syn(0, 0, dest_x, dest_y, *sp, DEFAULT_delay_weight, psi->deltaT, type);
        syn.W = weight * g_synapseStrengthAdjustmentConstant;
        synapseList.push_back(syn);
    }

    fSInput = true;
}

/**
 * Terminate process.
 */
void SInputPoisson::term()
{
}
