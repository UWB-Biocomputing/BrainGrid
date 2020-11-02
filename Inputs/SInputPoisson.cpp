/*
 *      \file SInputPoisson.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson).
 */

#include "SInputPoisson.h"
#include "AllDSSynapses.h"

/*
 * constructor
 * @param[in] psi       Pointer to the simulation information parms element
 * @param[in] fr_mean   Firing rate (per sec)
 * @param[in] weight    Synapse weight
 * @param[in] maskIndex Input masks index
 */
SInputPoisson::SInputPoisson(SimulationInfo* psi, BGFLOAT fr_mean, BGFLOAT weight, vector<BGFLOAT> &maskIndex) :
    SInput::SInput(psi, weight, maskIndex)
{
     // initialize firng rate, inverse firing rate
    fr_mean = fr_mean / 1000;	// firing rate per msec
    m_lambda = 1 / fr_mean;	// inverse firing rate

    m_fSInput = true;
}

/*
 * destructor
 */
SInputPoisson::~SInputPoisson()
{
}

/*
 * Initialize data.
 *
 *  @param[in] psi            Pointer to the simulation information.
 *  @param[in] vtClrInfo      Vector of ClusterInfo.
 */
void SInputPoisson::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (m_fSInput == false)
        return;

    SInput::init(psi, vtClrInfo);
}

/*
 * Terminate process.
 *
 *  @param[in] psi             Pointer to the simulation information.
 *  @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void SInputPoisson::term(SimulationInfo* psi, vector<ClusterInfo *> const &vtClrInfo)
{
    SInput::term(psi, vtClrInfo);
}
