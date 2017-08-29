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

    if (m_fSInput == false)
        return;
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputPoisson::term(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    SInputPoisson::term(psi, vtClrInfo);
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
    if (m_fSInput == false)
        return;

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++)
    {
        ClusterInfo *clr_info = vtClrInfo[iCluster];

        int neuronLayoutIndex = clr_info->clusterNeuronsBegin;
        int totalClusterNeurons = clr_info->totalClusterNeurons;

        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++)
        {
            if (m_masks[neuronLayoutIndex] == false)
                continue;

            BGSIZE iSyn = m_maxSynapsesPerNeuron * iNeuron;
            if (--m_nISIs[neuronLayoutIndex] <= 0)
            {
                // add a spike
                dynamic_cast<AllSpikingSynapses*>(clr_info->synapsesSInput)->preSpikeHit(iSyn, vtClrInfo[iCluster]->clusterID);

                // update interval counter (exponectially distribution ISIs, Poisson)
                BGFLOAT isi = -m_lambda * log(clr_info->rng->inRange(0, 1));
                // delete isi within refractoriness
                while (clr_info->rng->inRange(0, 1) <= exp(-(isi*isi)/32))
                    isi = -m_lambda * log(clr_info->rng->inRange(0, 1));
                // convert isi from msec to steps
                m_nISIs[neuronLayoutIndex] = static_cast<int>( (isi / 1000) / psi->deltaT + 0.5 );
            }

            // process synapse & apply psr to the summation point
            (clr_info->synapsesSInput)->advanceSynapse(iSyn, psi, NULL);
        }

        // Advances synapses pre spike event queue state of the cluster one simulation step
        dynamic_cast<AllSpikingSynapses*>(clr_info->synapsesSInput)->advanceSpikeQueue();
    }
}
