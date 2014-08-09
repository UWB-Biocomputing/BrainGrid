/**
 *      \file HostSInputRegular.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular).
 */

#include "HostSInputRegular.h"

/**
 * constructor
 */
HostSInputRegular::HostSInputRegular() : SInputRegular()
{
    
}

/**
 * destructor
 */
HostSInputRegular::~HostSInputRegular()
{
}

/**
 * Initialize data.
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] parms     Pointer to xml parms element.
 */
void HostSInputRegular::init(SimulationInfo* psi, TiXmlElement* parms)
{
    SInputRegular::init(psi, parms);
}

/**
 * Terminate process.
 */
void HostSInputRegular::term()
{
    if (values != NULL)
        delete[] values;

    if (nShiftValues != NULL)
        delete[] nShiftValues;
}

/**
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 * @param[in] summationPoint
 */
void HostSInputRegular::inputStimulus(SimulationInfo* psi, BGFLOAT* summationPoint)
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
            summationPoint[i] += values[i];
    }

    // update cycle count 
    nStepsInCycle = (nStepsInCycle + 1) % nStepsCycle;
}
