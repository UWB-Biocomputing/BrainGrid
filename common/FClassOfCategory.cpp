/*
 *      \file FClassOfCategory.cpp
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A factoy class for creating a class of the category.
 */

#include "FClassOfCategory.h"
#include "AllLIFNeurons.h"
#include "AllIZHNeurons.h"
#include "AllDSSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "ConnGrowth.h"
#include "ConnStatic.h"
#include "ParseParamError.h"

/*
 * constructor
 */
FClassOfCategory::FClassOfCategory() :
    m_neurons(NULL),
    m_synapses(NULL),
    m_conns(NULL),
    m_layout(NULL)
{
    // register neurons classes
    registerNeurons("AllLIFNeurons", &AllLIFNeurons::Create);
    registerNeurons("AllIZHNeurons", &AllIZHNeurons::Create);

    // register synapses classes
    registerSynapses("AllDSSynapses", &AllDSSynapses::Create);
    registerSynapses("AllSTDPSynapses", &AllSTDPSynapses::Create);
    registerSynapses("AllDynamicSTDPSynapses", &AllDynamicSTDPSynapses::Create);

    // register connections classes
    registerConns("ConnGrowth", &ConnGrowth::Create);
    registerConns("ConnStatic", &ConnStatic::Create);

    // register layout classes    
    registerLayout("Layout", &Layout::Create);
}

/*
 * destructor
 */
FClassOfCategory::~FClassOfCategory()
{
    m_FactoryMapNeurons.clear();
    m_FactoryMapSynapses.clear();
    m_FactoryMapConns.clear();
    m_FactoryMapLayout.clear();
}

void FClassOfCategory::registerNeurons(const string &neuronsClassName, CreateNeuronsFn pfnCreateNeurons)
{
    m_FactoryMapNeurons[neuronsClassName] = pfnCreateNeurons;
}

void FClassOfCategory::registerSynapses(const string &synapsesClassName, CreateSynapsesFn pfnCreateSynapses)
{
    m_FactoryMapSynapses[synapsesClassName] = pfnCreateSynapses;
}

void FClassOfCategory::registerConns(const string &connsClassName, CreateConnsFn pfnCreateConns)
{
    m_FactoryMapConns[connsClassName] = pfnCreateConns;
}

void FClassOfCategory::registerLayout(const string &layoutClassName, CreateLayoutFn pfnCreateLayout)
{
    m_FactoryMapLayout[layoutClassName] = pfnCreateLayout;
}

/**
 * Create an instance
 */
IAllNeurons* FClassOfCategory::createNeurons(TiXmlElement* parms)
{
    string neuronsClassName = "AllLIFNeurons";

    m_neurons = createNeuronsWithName(neuronsClassName);
    return m_neurons;
}

IAllSynapses* FClassOfCategory::createSynapses(TiXmlElement* parms)
{
    string synapsesClassName = "AllDSSynapses";

    m_synapses = createSynapsesWithName(synapsesClassName);
    return m_synapses;
}

Connections* FClassOfCategory::createConnections(TiXmlElement* parms)
{
    string connsClassName = "ConnGrowth";

    m_conns = createConnsWithName(connsClassName);
    return m_conns;
}

Layout* FClassOfCategory::createLayout(TiXmlElement* parms)
{
    string layoutClassName = "Layout";

    m_layout = createLayoutWithName(layoutClassName);
    return m_layout;
}

IAllNeurons* FClassOfCategory::createNeuronsWithName(const string& neuronsClassName) 
{
    FactoryMapNeurons::iterator it = m_FactoryMapNeurons.find(neuronsClassName);
    if (it != m_FactoryMapNeurons.end())
        return it->second();
    return NULL;
}

IAllSynapses* FClassOfCategory::createSynapsesWithName(const string& synapsesClassName)
{
    FactoryMapSynapses::iterator it = m_FactoryMapSynapses.find(synapsesClassName);
    if (it != m_FactoryMapSynapses.end())
        return it->second();
    return NULL;
}

Connections* FClassOfCategory::createConnsWithName(const string& connsClassName)
{
    FactoryMapConns::iterator it = m_FactoryMapConns.find(connsClassName);
    if (it != m_FactoryMapConns.end())
        return it->second();
    return NULL;
}

Layout* FClassOfCategory::createLayoutWithName(const string& layoutClassName)
{
    FactoryMapLayout::iterator it = m_FactoryMapLayout.find(layoutClassName);
    if (it != m_FactoryMapLayout.end())
        return it->second();
    return NULL;
}

/*
 *  Attempts to read parameters from a XML file.
 *  @param  source  the TiXmlElement to read from.
 *  @return true if successful, false otherwise.
 */
bool FClassOfCategory::readParameters(TiXmlElement *source)
{
    try {
         source->Accept(this);
    } catch (ParseParamError &error) {
        error.print(cerr);
        cerr << endl;
        return false;
    }

    return true;
}

/*
 *  Takes an XmlElement and checks for errors. If not, calls getValueList().
 *  @param  element TiXmlElement to examine.
 *  @param  firstAttribute  ***NOT USED***.
 *  @return true if method finishes without errors.
 */
bool FClassOfCategory::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
//TODO: firstAttribute does not seem to be used! Delete?
{
    // Read neurons parameters
    if (m_neurons->readParameters(element) != true) {
        throw ParseParamError("Neurons", "Failed in readParameters.");
    }

    // Read synapses parameters
    if (m_synapses->readParameters(element) != true) {
        throw ParseParamError("Synapses", "Failed in readParameters.");
    }

    // Read connections parameters (growth parameters)
    if (m_conns->readParameters(element) != true) {
        throw ParseParamError("Connections", "Failed in readParameters.");
    }

    // Read layout parameters
    if (m_layout->readParameters(element) != true) {
        throw ParseParamError("Layout", "Failed in readParameters.");
    }

    return true;
}

/*
 *  Prints out all parameters of the model to ostream.
 *  @param  output  ostream to send output to.
 */
void FClassOfCategory::printParameters(ostream &output) const
{
    // Prints all neurons parameters
    m_neurons->printParameters(output);

    // Prints all synapses parameters
    m_synapses->printParameters(output);

    // Prints all connections parameters
    m_conns->printParameters(output);

    // Prints all layout parameters
    m_layout->printParameters(output);
}
