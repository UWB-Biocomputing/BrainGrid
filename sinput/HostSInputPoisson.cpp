/**
 *      \file HostSInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "HostSInputPoisson.h"
#include "tinyxml/tinyxml.h"

/**
 * constructor
 */
HostSInputPoisson::HostSInputPoisson() : SInputPoisson()
{
    
}

/**
 * destructor
 */
HostSInputPoisson::~HostSInputPoisson()
{
}

/**
 * Initialize data.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] parms     Pointer to xml parms element
 */
void HostSInputPoisson::init(SimulationInfo* psi, TiXmlElement* parms)
{
    SInputPoisson::init(psi, parms);
}

/**
 * Terminate process.
 */
void HostSInputPoisson::term()
{
    SInputPoisson::term();

    // clear the synapse layer, which destroy all synase objects
    synapseList.clear();

    // clear memory for interval counter
    if (nISIs != NULL)
        delete[] nISIs;
}

/**
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 * @param[in] summationPoint
 */
void HostSInputPoisson::inputStimulus(SimulationInfo* psi, BGFLOAT* summationPoint)
{
    if (fSInput == false)
        return;

#if defined(USE_OMP)
int chunk_size = psi->cNeurons / omp_get_max_threads();
#endif

#if defined(USE_OMP)
#pragma omp parallel for schedule(static, chunk_size)
#endif
    for (int i = 0; i < psi->cNeurons; i++)
    {
        if (--nISIs[i] <= 0)
        {
            // add a spike
            synapseList[i].preSpikeHit();

            // update interval counter (exponectially distribution ISIs, Poisson)
            BGFLOAT isi = -lambda * log(rng.inRange(0, 1));
            // delete isi within refractoriness
            while (rng.inRange(0, 1) <= exp(-(isi*isi)/32))
                isi = -lambda * log(rng.inRange(0, 1));
            // convert isi from msec to steps
            nISIs[i] = static_cast<int>( (isi / 1000) / psi->deltaT + 0.5 );
        }
        // process synapse
        synapseList[i].advance();
    }
}
