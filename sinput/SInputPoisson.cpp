/**
 *      \file SInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "SInputPoisson.h"
#include "tinyxml.h"
#include "LIFSingleThreadedModel.h"

/**
 * constructor
 * @param[in] parms     Pointer to xml parms element
 */
SInputPoisson::SInputPoisson(SimulationInfo* psi, TiXmlElement* parms)
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
    
    fSInput = true;
}

/**
 * destructor
 */
SInputPoisson::~SInputPoisson()
{
}

/**
 * Initialize data.
 * @param[in] model	Pointer to the Neural Network Model object.
 * @param[in] neurons  	The Neuron list to search from.
 * @param[in] psi       Pointer to the simulation information.
 */
void SInputPoisson::init(Model* model, AllNeurons &neurons, SimulationInfo* psi)
{
    if (fSInput == false)
        return;

    // create an input synapse layer
    synapses = new AllSynapses(psi->totalNeurons, 1);
    for (int neuron_index = 0; neuron_index < psi->totalNeurons; neuron_index++)
    {
        int x = neuron_index % psi->width;
        int y = neuron_index / psi->width;
        Coordinate dest(x, y);
        synapseType type;
        if (neurons.neuron_type_map[neuron_index] == INH)
            type = EI;
        else
            type = EE;
        BGFLOAT* sum_point = &( psi->pSummationMap[neuron_index] );
        static_cast<LIFSingleThreadedModel*>(model)->createSynapse(*synapses, neuron_index, 0, NULL, dest, sum_point, psi->deltaT, type);
        synapses->W[neuron_index][0] = weight * LIFModel::SYNAPSE_STRENGTH_ADJUSTMENT;
    }
}

/**
 * Terminate process.
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 */
void SInputPoisson::term(Model* model, SimulationInfo* psi)
{
    // clear memory for interval counter
    if (nISIs != NULL)
        delete[] nISIs;

    // clear the synapse layer, which destroy all synase objects
    if (synapses != NULL)
        delete synapses;
}
