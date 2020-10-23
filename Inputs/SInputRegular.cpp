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
 * @param[in] duration     Duration of a pulse in second
 * @param[in] interval     Interval between pulses in second
 * @parm[in] sync          'yes, 'no', or 'wave'
 * @param[in] initValues   Initial input values
 */
SInputRegular::SInputRegular(SimulationInfo* psi, BGFLOAT duration, BGFLOAT interval, string &sync, vector<BGFLOAT> &initValues) :
    m_values(NULL),
    m_nShiftValues(NULL)
{
    m_fSInput = false;
    m_duration = duration;
    m_interval = interval;

    // initialize duration ,interval and cycle
    m_nStepsDuration = static_cast<int> ( m_duration / psi->deltaT + 0.5 );
    m_nStepsInterval = static_cast<int> ( m_interval / psi->deltaT + 0.5 );
    m_nStepsCycle = m_nStepsDuration + m_nStepsInterval;

    // allocate memory for input values
    m_values = new BGFLOAT[psi->totalNeurons];

    // initialize values
    for (int i = 0; i < psi->height; i++)
        for (int j = 0; j < psi->width; j++)
            m_values[i * psi->width + j] = initValues[(i % 10) * 10 + j % 10];

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
 * @param[in] psi       Pointer to the simulation information.
 */
void SInputRegular::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++)
    {
        vtClrInfo[iCluster]->nStepsInCycle = 0;
    }
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void SInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo)
{
}

/* 
 * Helper function for input vaue list (copied from BGDriver.cpp and modified for BGFLOAT)
 */
void getValueList(const string& valString, vector<BGFLOAT>* pList)
{
    std::istringstream valStream(valString);
    BGFLOAT i;

    // Parse integers out of the string and add them to a list
    while (valStream.good())
    {
        valStream >> i;
        pList->push_back(i);
    }
}

/*
 * Advance input stimulus state.
 *
 * @param[in] pci             ClusterInfo class to read information from.
 * @param[in] iStep           Simulation steps to advance.
 */
void SInputRegular::advanceSInputState(const ClusterInfo *pci, int iStep)
{
}
