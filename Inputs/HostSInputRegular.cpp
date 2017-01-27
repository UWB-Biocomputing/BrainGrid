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
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputRegular::init(SimulationInfo* psi, ClusterInfo* pci)
{
    SInputRegular::init(psi, pci);
}

/*
 * Terminate process.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputRegular::term(SimulationInfo* psi)
{
    if (values != NULL)
        delete[] values;

    if (nShiftValues != NULL)
        delete[] nShiftValues;
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi             Pointer to the simulation information.
 */
void HostSInputRegular::inputStimulus(const SimulationInfo* psi, const ClusterInfo *pci)
{
    if (fSInput == false)
        return;

    // add input to each summation point
    for (int i = pci->totalClusterNeurons - 1; i >= 0; --i)
    {
        if ( (nStepsInCycle >= nShiftValues[i]) && (nStepsInCycle < (nShiftValues[i] + nStepsDuration ) % nStepsCycle) )
            pci->pClusterSummationMap[i] += values[i];
    }

    // update cycle count 
    nStepsInCycle = (nStepsInCycle + 1) % nStepsCycle;
}
