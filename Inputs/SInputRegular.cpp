/*
 *      \file SInputRegular.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular).
 */

#include "SInputRegular.h"
#include "tinyxml.h"

void getValueList(const string& valString, vector<BGFLOAT>* pList);

/*
 * constructor
 *
 * @param[in] psi       Pointer to the simulation information
 * @param[in] parms     Pointer to xml parms element
 */
SInputRegular::SInputRegular(SimulationInfo* psi, TiXmlElement* parms) :
    values(NULL),
    nShiftValues(NULL)
{
    fSInput = false;

    // read duration, interval and sync
    TiXmlElement* temp = NULL;
    string sync;
    if (( temp = parms->FirstChildElement( "IntParams" ) ) != NULL) { if (temp->QueryFLOATAttribute("duration", &duration ) != TIXML_SUCCESS) {
            cerr << "error IntParams:duration" << endl;
            return;
        }
        if (temp->QueryFLOATAttribute("interval", &interval ) != TIXML_SUCCESS) {
            cerr << "error IntParams:interval" << endl;
            return;
        }
        if (temp->QueryValueAttribute("sync", &sync ) != TIXML_SUCCESS) {
            cerr << "error IntParams:sync" << endl;
            return;
        }
    }
    else
    {
        cerr << "missing IntParams" << endl;
        return;
    }

    // initialize duration ,interval and cycle
    nStepsDuration = static_cast<int> ( duration / psi->deltaT + 0.5 );
    nStepsInterval = static_cast<int> ( interval / psi->deltaT + 0.5 );
    nStepsCycle = nStepsDuration + nStepsInterval;
    nStepsInCycle = 0;

    // read initial values
    if ((temp = parms->FirstChildElement( "Values")) != NULL)
    {
        TiXmlNode* pNode = NULL;
        while ((pNode = temp->IterateChildren(pNode)) != NULL)
        {
            if (strcmp(pNode->Value(), "I") == 0)
            {
                getValueList(pNode->ToElement()->GetText(), &initValues);
            }
            else
            {
                cerr << "error I" << endl;
                return;
            }
        }
    }
    else
    {
        cerr << "missing Values" << endl;
        return;
    }

    // we assume that initial values are in 10x10 matrix
    assert(initValues.size() == 100);

    // allocate memory for input values
    values = new BGFLOAT[psi->totalNeurons];

    // initialize values
    for (int i = 0; i < psi->height; i++)
        for (int j = 0; j < psi->width; j++)
            values[i * psi->width + j] = initValues[(i % 10) * 10 + j % 10];

    initValues.clear();

    // allocate memory for shift values
    nShiftValues = new int[psi->totalNeurons];

    // initialize shift values
    memset(nShiftValues, 0, sizeof(int) * psi->totalNeurons);

    if (sync == "no")
    {
       // asynchronous stimuli - fill nShiftValues array with values between 0 - nStepsCycle
        for (int i = 0; i < psi->height; i++)
            for (int j = 0; j < psi->width; j++)
                nShiftValues[i * psi->width + j] = static_cast<int>(rng.inRange(0, nStepsCycle - 1));
    }
    else if (sync == "wave")
    {
        // wave stimuli - moving wave from left to right
        for (int i = 0; i < psi->height; i++)
            for (int j = 0; j < psi->width; j++)
                nShiftValues[i * psi->width + j] = static_cast<int>((nStepsCycle / psi->width) * j);
    }

    fSInput = true;
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
void SInputRegular::init(SimulationInfo* psi)
{
}

/*
 * Terminate process.
 *
 * @param[in] psi                Pointer to the simulation information.
 */
void SInputRegular::term(SimulationInfo* psi)
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
