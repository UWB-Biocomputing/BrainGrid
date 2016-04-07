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
void HostSInputRegular::init(SimulationInfo* psi)
{
    SInputRegular::init(psi);
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
void HostSInputRegular::inputStimulus(SimulationInfo* psi)
{
    if (fSInput == false)
        return;

#if defined(USE_OMP)
int chunk_size = psi->totalNeurons / omp_get_max_threads();
#endif

#if defined(USE_OMP)
#pragma omp parallel for schedule(static, chunk_size)
#endif
    // add input to each summation point
    for (int i = psi->totalNeurons - 1; i >= 0; --i)
    {
        if ( (nStepsInCycle >= nShiftValues[i]) && (nStepsInCycle < (nShiftValues[i] + nStepsDuration ) % nStepsCycle) )
            psi->pSummationMap[i] += values[i];
    }

    // update cycle count 
    nStepsInCycle = (nStepsInCycle + 1) % nStepsCycle;
}
