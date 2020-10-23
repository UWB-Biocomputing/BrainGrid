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
 * @param[in] duration     Duration of a pulse in second
 * @param[in] interval     Interval between pulses in second
 * @parm[in] sync          'yes, 'no', or 'wave'
 * @param[in] initValues   Initial input values
 */
HostSInputRegular::HostSInputRegular(SimulationInfo* psi, BGFLOAT duration, BGFLOAT interval, string &sync, vector<BGFLOAT> &initValues) : SInputRegular(psi, duration, interval, sync, initValues)
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
void HostSInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo)
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
 * @param[in] pci             ClusterInfo class to read information from.
 * @param[in] iStepOffset     Offset from the current simulation step.
 */
void HostSInputRegular::inputStimulus(const SimulationInfo* psi, ClusterInfo *pci, int iStepOffset)
{
    if (m_fSInput == false)
        return;

    int neuronLayoutIndex = pci->clusterNeuronsBegin;
    int totalClusterNeurons = pci->totalClusterNeurons;

    // add input to each summation point
    for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++) {
        if ( (pci->nStepsInCycle >= m_nShiftValues[neuronLayoutIndex]) && (pci->nStepsInCycle < (m_nShiftValues[neuronLayoutIndex] + m_nStepsDuration ) % m_nStepsCycle) )
            pci->pClusterSummationMap[iNeuron] += m_values[neuronLayoutIndex];
    }

    // update cycle count 
    pci->nStepsInCycle = (pci->nStepsInCycle + 1) % m_nStepsCycle;
}

