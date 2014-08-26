/**
 *      \file SInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "SInputPoisson.h"
#include "tinyxml.h"

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
 * @param[in] model	Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] parms     Pointer to xml parms element
 */
void SInputPoisson::init(Model* model, SimulationInfo* psi, TiXmlElement* parms)
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
 * Terminate process.
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 */
void SInputPoisson::term(Model* model, SimulationInfo* psi)
{
    // clear memory for interval counter
    if (nISIs != NULL)
        delete[] nISIs;
}
