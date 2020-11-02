/*
 *      \file HostSInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "HostSInputPoisson.h"
#include "tinyxml.h"

/*
 * The constructor for HostSInputPoisson.
 *
 * @param[in] psi       Pointer to the simulation information
 * @param[in] fr_mean   Firing rate (per sec)
 * @param[in] weight    Synapse weight
 * @param[in] maskIndex Input masks index
 */
HostSInputPoisson::HostSInputPoisson(SimulationInfo* psi, BGFLOAT fr_mean, BGFLOAT weight, vector<BGFLOAT> &maskIndex) : SInputPoisson(psi, fr_mean, weight, maskIndex)
{
    
}

/*
 * destructor
 */
HostSInputPoisson::~HostSInputPoisson()
{
}

/*
 * Initialize data.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputPoisson::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    SInputPoisson::init(psi, vtClrInfo);
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputPoisson::term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo)
{
    SInputPoisson::term(psi, vtClrInfo);
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] pci             ClusterInfo class to read information from.
 * @param[in] iStepOffset     Offset from the current simulation step.
 */
void HostSInputPoisson::inputStimulus(const SimulationInfo* psi, ClusterInfo *pci, int iStepOffset)
{
    if (m_fSInput == false)
        return;

    AllSynapses *pSynapses = dynamic_cast<AllSynapses*>(pci->synapsesSInput);
    int maxSpikes = (int) ((psi->epochDuration * psi->maxFiringRate));
    uint64_t simulationStep = g_simulationStep + iStepOffset;
    AllSynapsesProps* pSynapsesProps = pSynapses->m_pSynapsesProps;

    int neuronLayoutIndex = pci->clusterNeuronsBegin;
    int totalClusterNeurons = pci->totalClusterNeurons;

    for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++)
    {
        if (m_masks[neuronLayoutIndex] == false)
            continue;

        BGSIZE iSyn = m_maxSynapsesPerNeuron * iNeuron;
        if (--m_nISIs[neuronLayoutIndex] <= 0)
        {
            // add a spike
            dynamic_cast<AllSpikingSynapses*>(pci->synapsesSInput)->preSpikeHit(iSyn, pci->clusterID, iStepOffset);

            // update interval counter (exponectially distribution ISIs, Poisson)
            BGFLOAT isi = -m_lambda * log(pci->rng->inRange(0, 1));
            // delete isi within refractoriness
            while (pci->rng->inRange(0, 1) <= exp(-(isi*isi)/32))
                isi = -m_lambda * log(pci->rng->inRange(0, 1));
            // convert isi from msec to steps
            m_nISIs[neuronLayoutIndex] = static_cast<int>( (isi / 1000) / psi->deltaT + 0.5 );
        }

        // process synapse & apply psr to the summation point
        pSynapses->advanceSynapse(iSyn, psi->deltaT, NULL, simulationStep, iStepOffset, maxSpikes, NULL);

        BGFLOAT &summationPoint = *(pSynapsesProps->summationPoint[iSyn]);
        BGFLOAT &psr = pSynapsesProps->psr[iSyn];
        summationPoint += psr;
    }
}
