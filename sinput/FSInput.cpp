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
#if defined(USE_GPU)
        pInput = new GpuSInputRegular(psi, parms);
#else
        pInput = new HostSInputRegular(psi, parms);
#endif
    }
    else if (name == "SInputPoisson")
    {
#if defined(USE_GPU)
        pInput = new GpuSInputPoisson(psi, parms);
#else
        pInput = new HostSInputPoisson(psi, parms);
#endif
    }
    else
    {
        cerr << "unsupported stimulus input method" << endl;
    }

    return pInput;
}

