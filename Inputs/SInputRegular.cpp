/*
 *      \file SInputRegular.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular).
 */

#include "SInputRegular.h"

/*
 * constructor
 *
 * @param[in] psi          Pointer to the simulation information
 * @param[in] firingRate   Firing Rate (Hz)
 * @param[in] duration     Duration of a pulse in second
 * @param[in] interval     Interval between pulses in second
 * @param[in] sync         'yes, 'no', or 'wave'
 * @param[in] weight       Synapse weight
 * @param[in] maskIndex    Input masks index
 */
SInputRegular::SInputRegular(SimulationInfo* psi, BGFLOAT firingRate, BGFLOAT duration, BGFLOAT interval, string const &sync, BGFLOAT weight, vector<BGFLOAT> const &maskIndex) :
    SInput::SInput(psi, weight, maskIndex),
    m_nShiftValues(NULL)
{
    m_duration = duration;
    m_interval = interval;
    BGFLOAT isi = 1.0 / firingRate; // interval in second
    m_nISI = static_cast<int>( isi / psi->deltaT + 0.5 ); // convert isi from sec to steps

    // initialize duration ,interval and cycle
    m_nStepsDuration = static_cast<int> ( m_duration / psi->deltaT + 0.5 );
    m_nStepsInterval = static_cast<int> ( m_interval / psi->deltaT + 0.5 );
    m_nStepsCycle = m_nStepsDuration + m_nStepsInterval;

    // allocate memory for shift values
    m_nShiftValues = new int[psi->totalNeurons];

    // initialize shift values
    memset(m_nShiftValues, 0, sizeof(int) * psi->totalNeurons);

    if (sync == "no")
    {
       // asynchronous stimuli - fill nShiftValues array with values between 0 - nStepsCycle
        for (int i = 0; i < psi->height; i++)
            for (int j = 0; j < psi->width; j++)
                m_nShiftValues[i * psi->width + j] = static_cast<int>(rng.inRange(0, m_nStepsCycle - 1));
    }
    else if (sync == "wave")
    {
        // wave stimuli - moving wave from left to right
        for (int i = 0; i < psi->height; i++)
            for (int j = 0; j < psi->width; j++)
                m_nShiftValues[i * psi->width + j] = static_cast<int>((m_nStepsCycle / psi->width) * j);
    }

    m_fSInput = true;
}

/*
 * destructor
 */
SInputRegular::~SInputRegular()
{
}

/*
 * Initialize data.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void SInputRegular::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (m_fSInput == false)
        return;

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++)
    {
        vtClrInfo[iCluster]->nStepsInCycle = 0;
    }

    SInput::init(psi, vtClrInfo);
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void SInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> const &vtClrInfo)
{
    // clear memory for shift values
    if (m_nShiftValues != NULL)
        delete[] m_nShiftValues;

    SInput::term(psi, vtClrInfo);
}

