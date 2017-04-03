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
 * @param[in] parms     TiXmlElement to examine.
 */
HostSInputPoisson::HostSInputPoisson(SimulationInfo* psi, TiXmlElement* parms) : SInputPoisson(psi, parms)
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

    if (fSInput == false)
        return;
}

/*
 * Terminate process.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputPoisson::term(SimulationInfo* psi)
{
    SInputPoisson::term(psi);
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputPoisson::inputStimulus(const SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (fSInput == false)
        return;

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++)
    {
        int neuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin;
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;

        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++)
        {
            if (masks[neuronLayoutIndex] == false)
                continue;

            BGSIZE iSyn = m_maxSynapsesPerNeuron * neuronLayoutIndex;
            if (--nISIs[neuronLayoutIndex] <= 0)
            {
                // add a spike
                dynamic_cast<AllSpikingSynapses*>(m_synapses)->preSpikeHit(iSyn, m_clusterInfo->clusterID);

                // update interval counter (exponectially distribution ISIs, Poisson)
                BGFLOAT isi = -lambda * log(rng.inRange(0, 1));
                // delete isi within refractoriness
                while (rng.inRange(0, 1) <= exp(-(isi*isi)/32))
                    isi = -lambda * log(rng.inRange(0, 1));
                // convert isi from msec to steps
                nISIs[neuronLayoutIndex] = static_cast<int>( (isi / 1000) / psi->deltaT + 0.5 );
            }

            // process synapse
            m_synapses->advanceSynapse(iSyn, psi, NULL);
        }
    }

    // Advances synapses pre spike event queue state of the cluster one simulation step
    dynamic_cast<AllSpikingSynapses*>(m_synapses)->advanceSpikeQueue();
}
