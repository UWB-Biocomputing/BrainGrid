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
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] neurons   The Neuron list to search from.
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputRegular::init(IModel* model, IAllNeurons &neurons, SimulationInfo* psi)
{
    SInputRegular::init(model, neurons, psi);
}

/*
 * Terminate process.
 *
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] psi       Pointer to the simulation information.
 */
void HostSInputRegular::term(IModel* model, SimulationInfo* psi)
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
 * @param[in] model           Pointer to the Neural Network Model object.
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] summationPoint  Poiner to the summation point.
 */
void HostSInputRegular::inputStimulus(IModel* model, SimulationInfo* psi, BGFLOAT* summationPoint)
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
