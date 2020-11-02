/*
 *      \file FSInput.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A factoy class that creates an instance of stimulus input object.
 */

#include "FSInput.h"
#include "HostSInputRegular.h"
#include "HostSInputPoisson.h"
#if defined(USE_GPU)
#include "GpuSInputRegular.h"
#include "GpuSInputPoisson.h"
#endif
#include "tinyxml.h"

void getValueList(const string& valString, vector<BGFLOAT>* pList);

/*
 * constructor
 */
FSInput::FSInput()
{
    
}

/*
 * destructor
 */
FSInput::~FSInput()
{
}

/*
 * Create an instance of the stimulus input class based on the method
 * specified in the stimulus input file.
 *
 * @param[in] psi                   Pointer to the simulation information
 * @return a pointer to a SInput object
 */
ISInput* FSInput::CreateInstance(SimulationInfo* psi)
{
    if (psi->stimulusInputFileName.empty())
    {
        return NULL;
    }

    // load stimulus input file
    TiXmlDocument siDoc( psi->stimulusInputFileName.c_str( ) );
    if (!siDoc.LoadFile( )) 
    {
        cerr << "Failed loading stimulus input file " << psi->stimulusInputFileName << ":" << "\n\t"
                << siDoc.ErrorDesc( ) << endl;
        cerr << " error: " << siDoc.ErrorRow( ) << ", " << siDoc.ErrorCol( ) << endl;
        return NULL;
    }

    // load input parameters
    TiXmlElement* parms = NULL;
    if (( parms = siDoc.FirstChildElement( "InputParams" ) ) == NULL) 
    {
        cerr << "Could not find <InputParms> in stimulus input file " << psi->stimulusInputFileName << endl;
        return NULL;
    }

    // read input method
    TiXmlElement* temp = NULL;
    string name;
    if (( temp = parms->FirstChildElement( "IMethod" ) ) != NULL) { 
        if (temp->QueryValueAttribute("name", &name ) != TIXML_SUCCESS) {
            cerr << "error IMethod:name" << endl;
            return NULL;
        }
    }
    else
    {
        cerr << "missing IMethod" << endl;
        return NULL;
    }

    // create an instance
    ISInput* pInput = NULL;     // pointer to a stimulus input object

    if (name == "SInputRegular")
    {
        // read duration, interval and sync
        BGFLOAT duration;    // duration of a pulse in second
        BGFLOAT interval;    // interval between pulses in second
        string sync;         // synchnonous flag - 'yes', 'no' or 'wave'
        BGFLOAT weight;      // synapse weight
        BGFLOAT firingRate;  // firing rate (Hz)

        if (( temp = parms->FirstChildElement( "IntParams" ) ) != NULL) 
        { 
            if (temp->QueryFLOATAttribute("duration", &duration ) != TIXML_SUCCESS) {
                cerr << "error IntParams:duration" << endl;
                return NULL;
            }
            if (temp->QueryFLOATAttribute("interval", &interval ) != TIXML_SUCCESS) {
                cerr << "error IntParams:interval" << endl;
                return NULL;
            }
            if (temp->QueryValueAttribute("sync", &sync ) != TIXML_SUCCESS) {
                cerr << "error IntParams:sync" << endl;
                return NULL;
            }
            if (temp->QueryFLOATAttribute("weight", &weight ) != TIXML_SUCCESS) {
                cerr << "error IntParams:weight" << endl;
                return NULL;
            }
            if (temp->QueryFLOATAttribute("firingRate", &firingRate ) != TIXML_SUCCESS) {
                cerr << "error IntParams:firingRate" << endl;
                return NULL;
            }
        }
        else
        {
            cerr << "missing IntParams" << endl;
            return NULL;
        }

        // read mask values and set it to masks
        vector<BGFLOAT> maskIndex;
        if ((temp = parms->FirstChildElement( "Masks")) != NULL)
        {
           TiXmlNode* pNode = NULL;
            while ((pNode = temp->IterateChildren(pNode)) != NULL)
            {
                if (strcmp(pNode->Value(), "M") == 0)
                {
                    getValueList(pNode->ToElement()->GetText(), &maskIndex);
                }
                else if (strcmp(pNode->Value(), "LayoutFiles") == 0)
                {
                    string maskNListFileName;

                    if (pNode->ToElement()->QueryValueAttribute( "maskNListFileName", &maskNListFileName ) == TIXML_SUCCESS)
                    {
                        TiXmlDocument simDoc( maskNListFileName.c_str( ) );
                        if (!simDoc.LoadFile( ))
                        {
                            cerr << "Failed loading positions of stimulus input mask neurons list file " << maskNListFileName << ":" << "\n\t"
                                 << simDoc.ErrorDesc( ) << endl;
                            cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                            break;
                        }
                        TiXmlNode* temp2 = NULL;
                        if (( temp2 = simDoc.FirstChildElement( "M" ) ) == NULL)
                        {
                            cerr << "Could not find <M> in positons of stimulus input mask neurons list file " << maskNListFileName << endl;
                            break;
                        }
                        getValueList(temp2->ToElement()->GetText(), &maskIndex);
                    }
                }
            }
        }

#if defined(USE_GPU)
        pInput = new GpuSInputRegular(psi, firingRate, duration, interval, sync, weight, maskIndex);
#else
        pInput = new HostSInputRegular(psi, firingRate, duration, interval, sync, weight, maskIndex);
#endif
    }
    else if (name == "SInputPoisson")
    {
        BGFLOAT fr_mean;	// firing rate (per sec)
        BGFLOAT weight;		// synapse weight

        if (( temp = parms->FirstChildElement( "IntParams" ) ) != NULL)
        {
            if (temp->QueryFLOATAttribute("fr_mean", &fr_mean ) != TIXML_SUCCESS) {
                cerr << "error IntParams:fr_mean" << endl;
                return NULL;
            }
            if (temp->QueryFLOATAttribute("weight", &weight ) != TIXML_SUCCESS) {
                cerr << "error IntParams:weight" << endl;
                return NULL;
            }
        }
        else
        {
            cerr << "missing IntParams" << endl;
            return NULL;
        }

        // read mask values and set it to masks
        vector<BGFLOAT> maskIndex;
        if ((temp = parms->FirstChildElement( "Masks")) != NULL)
        {
           TiXmlNode* pNode = NULL;
            while ((pNode = temp->IterateChildren(pNode)) != NULL)
            {
                if (strcmp(pNode->Value(), "M") == 0)
                {
                    getValueList(pNode->ToElement()->GetText(), &maskIndex);
                }
                else if (strcmp(pNode->Value(), "LayoutFiles") == 0)
                {
                    string maskNListFileName;

                    if (pNode->ToElement()->QueryValueAttribute( "maskNListFileName", &maskNListFileName ) == TIXML_SUCCESS)
                    {
                        TiXmlDocument simDoc( maskNListFileName.c_str( ) );
                        if (!simDoc.LoadFile( ))
                        {
                            cerr << "Failed loading positions of stimulus input mask neurons list file " << maskNListFileName << ":" << "\n\t"
                                 << simDoc.ErrorDesc( ) << endl;
                            cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
                            break;
                        }
                        TiXmlNode* temp2 = NULL;
                        if (( temp2 = simDoc.FirstChildElement( "M" ) ) == NULL)
                        {
                            cerr << "Could not find <M> in positons of stimulus input mask neurons list file " << maskNListFileName << endl;
                            break;
                        }
                        getValueList(temp2->ToElement()->GetText(), &maskIndex);
                    }
                }
            }
        }

        // create a stimulus input object
#if defined(USE_GPU)
        pInput = new GpuSInputPoisson(psi, fr_mean, weight, maskIndex);
#else
        pInput = new HostSInputPoisson(psi, fr_mean, weight, maskIndex);
#endif
    }
    else
    {
        cerr << "unsupported stimulus input method" << endl;
    }

    return pInput;
}

/* 
 *  * Helper function for input vaue list (copied from BGDriver.cpp and modified for BGFLOAT)
 *   */
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
