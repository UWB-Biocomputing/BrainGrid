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
 * @param[in] psi       Pointer to the simulation information
 * @param[in] parms     TiXmlElement to examine.
 */
HostSInputRegular::HostSInputRegular(SimulationInfo* psi, TiXmlElement* parms) : SInputRegular(psi, parms)
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
void HostSInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (m_values != NULL)
        delete[] m_values;

    if (m_nShiftValues != NULL)
        delete[] m_nShiftValues;
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void HostSInputRegular::inputStimulus(const SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (m_fSInput == false)
        return;

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++) {
        int neuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin;
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;

        // add input to each summation point
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++) {
            if ( (m_nStepsInCycle >= m_nShiftValues[neuronLayoutIndex]) && (m_nStepsInCycle < (m_nShiftValues[neuronLayoutIndex] + m_nStepsDuration ) % m_nStepsCycle) )
                vtClrInfo[iCluster]->pClusterSummationMap[iNeuron] += m_values[neuronLayoutIndex];
        }
    }

    // update cycle count 
    m_nStepsInCycle = (m_nStepsInCycle + 1) % m_nStepsCycle;
}
