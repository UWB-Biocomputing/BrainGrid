/*
 *      \file HostSInputRegular.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular).
 */

#include "HostSInputRegular.h"

/*
 * constructor
 *
 * @param[in] psi          Pointer to the simulation information
 * @param[in] firingRate   Firing rate (Hz)
 * @param[in] duration     Duration of a pulse in second
 * @param[in] interval     Interval between pulses in second
 * @param[in] sync         'yes, 'no', or 'wave'
 * @param[in] weight       Synapse weight
 * @param[in] maskIndex    Input masks index
 */
HostSInputRegular::HostSInputRegular(SimulationInfo* psi, BGFLOAT firingRate, BGFLOAT duration, BGFLOAT interval, string const &sync, BGFLOAT weight, vector<BGFLOAT> const &maskIndex) : SInputRegular(psi, firingRate, duration, interval, sync, weight, maskIndex)
{
}

/**
 * destructor
 */
HostSInputRegular::~HostSInputRegular()
{
}

/*
 * Initialize data.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputRegular::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    SInputRegular::init(psi, vtClrInfo);
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> const &vtClrInfo)
{
    SInputRegular::term(psi, vtClrInfo);
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] pci             ClusterInfo class to read information from.
 * @param[in] iStepOffset     Offset from the current simulation step.
 */
void HostSInputRegular::inputStimulus(const SimulationInfo* psi, ClusterInfo *pci, int iStepOffset)
{
    if (m_fSInput == false)
        return;

    AllSynapses *pSynapses = dynamic_cast<AllSynapses*>(pci->synapsesSInput);
    int maxSpikes = (int) ((psi->epochDuration * psi->maxFiringRate));
    uint64_t simulationStep = g_simulationStep + iStepOffset;
    AllSynapsesProps* pSynapsesProps = pSynapses->m_pSynapsesProps;

    int neuronLayoutIndex = pci->clusterNeuronsBegin;
    int totalClusterNeurons = pci->totalClusterNeurons;

    // add input to each summation point
    for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++) {
        if (m_masks[neuronLayoutIndex] == false)
            continue;

        BGSIZE iSyn = m_maxSynapsesPerNeuron * iNeuron;
        if ( (pci->nStepsInCycle >= m_nShiftValues[neuronLayoutIndex]) && (pci->nStepsInCycle < (m_nShiftValues[neuronLayoutIndex] + m_nStepsDuration ) % m_nStepsCycle) )
        {
            if (--m_nISIs[neuronLayoutIndex] <= 0)
            {
                // add a spike
                dynamic_cast<AllSpikingSynapses*>(pci->synapsesSInput)->preSpikeHit(iSyn, pci->clusterID, iStepOffset);

                // set spikes interval
                m_nISIs[neuronLayoutIndex] = m_nISI;
            }
        }
        else 
        {
            m_nISIs[neuronLayoutIndex] = 0;
        }

        // process synapse & apply psr to the summation point
        pSynapses->advanceSynapse(iSyn, psi->deltaT, NULL, simulationStep, iStepOffset, maxSpikes, NULL);

        BGFLOAT &summationPoint = *(pSynapsesProps->summationPoint[iSyn]);
        BGFLOAT &psr = pSynapsesProps->psr[iSyn];
        summationPoint += psr;
    }

    // update cycle count 
    pci->nStepsInCycle = (pci->nStepsInCycle + 1) % m_nStepsCycle;
}

