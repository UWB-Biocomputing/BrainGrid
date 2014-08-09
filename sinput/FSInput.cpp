/**
 *      \file FSInput.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A factoy class that creates an instance of stimulus input object.
 */

#include "FSInput.h"
#include "HostSInputRegular.h"
#include "GpuSInputRegular.h"
//#include "HostSInputPoisson.h"
//#include "GpuSInputPoisson.h"
#include "tinyxml.h"

/**
 * constructor
 */
FSInput::FSInput()
{
    
}

/**
 * destructor
 */
FSInput::~FSInput()
{
}

/**
 * Create an instance
 * @param[in] psi       Pointer to the simulation information
 * @param[in] stimulusInputFileName Stimulus input file name
 * @return a pointer to a SInput object
 */
ISInput* FSInput::CreateInstance(SimulationInfo* psi, string stimulusInputFileName)
{
    if (stimulusInputFileName.empty())
    {
        return NULL;
    }

    // load stimulus input file
    TiXmlDocument siDoc( stimulusInputFileName.c_str( ) );
    if (!siDoc.LoadFile( )) 
    {
        cerr << "Failed loading stimulus input file " << stimulusInputFileName << ":" << "\n\t"
                << siDoc.ErrorDesc( ) << endl;
        cerr << " error: " << siDoc.ErrorRow( ) << ", " << siDoc.ErrorCol( ) << endl;
        return NULL;
    }

    // load input parameters
    TiXmlElement* parms = NULL;
    if (( parms = siDoc.FirstChildElement( "InputParams" ) ) == NULL) 
    {
        cerr << "Could not find <InputParms> in stimulus input file " << stimulusInputFileName << endl;
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
#if defined(USE_GPU)
        pInput = new GpuSInputRegular();
#else
        pInput = new HostSInputRegular();
#endif
        pInput->init(psi, parms);
    }
#if 0
    else if (name == "SInputPoisson")
    {
#if defined(USE_GPU)
        pInput = new GpuSInputPoisson();
#else
        pInput = new HostSInputPoisson();
#endif
        pInput->init(psi, parms);
    }
#endif
    else
    {
        cerr << "unsupported stimulus input method" << endl;
    }

    return pInput;
}

