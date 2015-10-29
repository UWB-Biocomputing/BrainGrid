/**
 *      \file HostSInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "HostSInputPoisson.h"
#include "SingleThreadedSpikingModel.h"
#include "tinyxml.h"

/**
 * constructor
 */
HostSInputPoisson::HostSInputPoisson(SimulationInfo* psi, TiXmlElement* parms) : SInputPoisson(psi, parms)
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
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] neurons   The Neuron list to search from.
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputPoisson::init(IModel* model, AllNeurons &neurons, SimulationInfo* psi)
{
    SInputPoisson::init(model, neurons, psi);

    if (fSInput == false)
        return;
}

/**
 * Terminate process.
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputPoisson::term(IModel* model, SimulationInfo* psi)
{
    SInputPoisson::term(model, psi);
}

/**
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] summationPoint
 */
void HostSInputPoisson::inputStimulus(IModel* model, SimulationInfo* psi, BGFLOAT* summationPoint)
{
    if (fSInput == false)
        return;

#if defined(USE_OMP)
int chunk_size = psi->totalNeurons / omp_get_max_threads();
#endif

#if defined(USE_OMP)
#pragma omp parallel for schedule(static, chunk_size)
#endif
    for (int neuron_index = 0; neuron_index < psi->totalNeurons; neuron_index++)
    {
        if (masks[neuron_index] == false)
            continue;

        uint32_t iSyn = psi->maxSynapsesPerNeuron * neuron_index;
        if (--nISIs[neuron_index] <= 0)
        {
            // add a spike
            synapses->preSpikeHit(iSyn);

            // update interval counter (exponectially distribution ISIs, Poisson)
            BGFLOAT isi = -lambda * log(rng.inRange(0, 1));
            // delete isi within refractoriness
            while (rng.inRange(0, 1) <= exp(-(isi*isi)/32))
                isi = -lambda * log(rng.inRange(0, 1));
            // convert isi from msec to steps
            nISIs[neuron_index] = static_cast<int>( (isi / 1000) / psi->deltaT + 0.5 );
        }
        // process synapse
        synapses->advanceSynapse(iSyn, psi, NULL);
    }
}
