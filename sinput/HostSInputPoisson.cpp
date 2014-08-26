/**
 *      \file HostSInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "HostSInputPoisson.h"
#include "LIFSingleThreadedModel.h"
#include "tinyxml.h"

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
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] parms     Pointer to xml parms element
 */
void HostSInputPoisson::init(Model* model, SimulationInfo* psi, TiXmlElement* parms)
{
    SInputPoisson::init(model, psi, parms);

    if (fSInput == false)
        return;

    // create an input synapse layer
    synapses = new AllSynapses(psi->totalNeurons, 1);
    for (int neuron_index = 0; neuron_index < psi->totalNeurons; neuron_index++)
    {
        int x = neuron_index % psi->width;
        int y = neuron_index / psi->width;
        Coordinate dest(x, y);
        synapseType type = EE;
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
void HostSInputPoisson::term(Model* model, SimulationInfo* psi)
{
    SInputPoisson::term(model, psi);

    // clear the synapse layer, which destroy all synase objects
    delete synapses;
}

/**
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] summationPoint
 */
void HostSInputPoisson::inputStimulus(Model* model, SimulationInfo* psi, BGFLOAT* summationPoint)
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
        if (--nISIs[neuron_index] <= 0)
        {
            // add a spike
            static_cast<LIFSingleThreadedModel*>(model)->preSpikeHit(*synapses, neuron_index, 0);

            // update interval counter (exponectially distribution ISIs, Poisson)
            BGFLOAT isi = -lambda * log(rng.inRange(0, 1));
            // delete isi within refractoriness
            while (rng.inRange(0, 1) <= exp(-(isi*isi)/32))
                isi = -lambda * log(rng.inRange(0, 1));
            // convert isi from msec to steps
            nISIs[neuron_index] = static_cast<int>( (isi / 1000) / psi->deltaT + 0.5 );
        }
        // process synapse
        static_cast<LIFSingleThreadedModel*>(model)->advanceSynapse(*synapses, neuron_index, 0, psi->deltaT);
    }
}
