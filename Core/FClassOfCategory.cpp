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
#include "FixedLayout.h"
#include "DynamicLayout.h"
#include "ParseParamError.h"
#include <typeinfo>

// Part of the stopgap approach for selecting model types, until parameter file selection
// is implemented. Used to convert a preprocessor defined symbol into a C string (i.e.,
// put quotes around it).
#define xstr(s) str(s)
#define str(s) #s

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
    registerSynapses("AllSpikingSynapses", &AllSpikingSynapses::Create);
    registerSynapses("AllDSSynapses", &AllDSSynapses::Create);
    registerSynapses("AllSTDPSynapses", &AllSTDPSynapses::Create);
    registerSynapses("AllDynamicSTDPSynapses", &AllDynamicSTDPSynapses::Create);

    // register connections classes
    registerConns("ConnGrowth", &ConnGrowth::Create);
    registerConns("ConnStatic", &ConnStatic::Create);

    // register layout classes    
    registerLayout("FixedLayout", &FixedLayout::Create);
    registerLayout("DynamicLayout", &DynamicLayout::Create);
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

/*
 *  Register neurons class and its creation function to the factory.
 *
 *  @param  neuronsClassName  neurons class name.
 *  @param  pfnCreateNeurons  Pointer to the class creation function.
 */
void FClassOfCategory::registerNeurons(const string &neuronsClassName, CreateNeuronsFn pfnCreateNeurons)
{
    m_FactoryMapNeurons[neuronsClassName] = pfnCreateNeurons;
}

/*
 *  Register synapses class and its creation function to the factory.
 *
 *  @param  synapsesClassName synapses class name.
 *  @param  pfnCreateNeurons  Pointer to the class creation function.
 */
void FClassOfCategory::registerSynapses(const string &synapsesClassName, CreateSynapsesFn pfnCreateSynapses)
{
    m_FactoryMapSynapses[synapsesClassName] = pfnCreateSynapses;
}

/*
 *  Register connections class and its creation function to the factory.
 *
 *  @param  connsClassName    connections class name.
 *  @param  pfnCreateNeurons  Pointer to the class creation function.
 */
void FClassOfCategory::registerConns(const string &connsClassName, CreateConnsFn pfnCreateConns)
{
    m_FactoryMapConns[connsClassName] = pfnCreateConns;
}

/*
 *  Register layout class and its creation function to the factory.
 *
 *  @param  layoutClassName   layout class name.
 *  @param  pfnCreateNeurons  Pointer to the class creation function.
 */
void FClassOfCategory::registerLayout(const string &layoutClassName, CreateLayoutFn pfnCreateLayout)
{
    m_FactoryMapLayout[layoutClassName] = pfnCreateLayout;
}

/*
 * Create an instance of the neurons class, which is specified in the parameter file.
 *
 * @param  element TiXmlNode to examine.
 * @return Poiner to the neurons object.
 */
IAllNeurons* FClassOfCategory::createNeurons(const TiXmlNode* parms)
{
    string neuronsClassName;
 
    if (parms->ToElement()->QueryValueAttribute("class", &neuronsClassName) == TIXML_SUCCESS) {
        m_neurons = createNeuronsWithName(neuronsClassName);
        return m_neurons;
    }

    return NULL;
}

/*
 * Create an instance of the synapses class, which is specified in the parameter file.
 *
 * @param  element TiXmlNode to examine.
 * @return Poiner to the synapses object.
 */
IAllSynapses* FClassOfCategory::createSynapses(const TiXmlNode* parms)
{
    string synapsesClassName;

    if (parms->ToElement()->QueryValueAttribute("class", &synapsesClassName) == TIXML_SUCCESS) {
        m_synapses = createSynapsesWithName(synapsesClassName);
        return m_synapses;
    }

    return NULL;
}

/*
 * Create an instance of the connections class, which is specified in the parameter file.
 *
 * @param  element TiXmlNode to examine.
 * @return Poiner to the connections object.
 */
Connections* FClassOfCategory::createConnections(const TiXmlNode* parms)
{
    string connsClassName;

    if (parms->ToElement()->QueryValueAttribute("class", &connsClassName) == TIXML_SUCCESS) {
        m_conns = createConnsWithName(connsClassName);
        return m_conns;
    }

    return NULL;
}

/*
 * Create an instance of the layout class, which is specified in the parameter file.
 *
 * @param  element TiXmlNode to examine.
 * @return Poiner to the layout object.
 */
Layout* FClassOfCategory::createLayout(const TiXmlNode* parms)
{
    string layoutClassName;

    if (parms->ToElement()->QueryValueAttribute("class", &layoutClassName) == TIXML_SUCCESS) {
        m_layout = createLayoutWithName(layoutClassName);
        return m_layout;
    }

    return NULL;
}

/*
 * Create an instance of the neurons class, which name is specified by neuronsClassName.
 *
 * @param  neuronsClassName neurons class name to create.
 * @return Poiner to the neurons object.
 */
IAllNeurons* FClassOfCategory::createNeuronsWithName(const string& neuronsClassName) 
{
    FactoryMapNeurons::iterator it = m_FactoryMapNeurons.find(neuronsClassName);
    if (it != m_FactoryMapNeurons.end())
        return it->second();
    return NULL;
}

/*
 * Create an instance of the synapses class, which name is specified by synapsesClassName.
 *
 * @param  synapsesClassName synapses class name to create.
 * @return Poiner to the synapses object.
 */
IAllSynapses* FClassOfCategory::createSynapsesWithName(const string& synapsesClassName)
{
    FactoryMapSynapses::iterator it = m_FactoryMapSynapses.find(synapsesClassName);
    if (it != m_FactoryMapSynapses.end())
        return it->second();
    return NULL;
}

/*
 * Create an instance of the connections class, which name is specified by connsClassName.
 *
 * @param  connsClassName connections class name to create.
 * @return Poiner to the connections object.
 */
Connections* FClassOfCategory::createConnsWithName(const string& connsClassName)
{
    FactoryMapConns::iterator it = m_FactoryMapConns.find(connsClassName);
    if (it != m_FactoryMapConns.end())
        return it->second();
    return NULL;
}

/*
 * Create an instance of the layout class, which name is specified by layoutClassName.
 *
 * @param  layoutClassName layout class name to create.
 * @return Poiner to the layout object.
 */
Layout* FClassOfCategory::createLayoutWithName(const string& layoutClassName)
{
    FactoryMapLayout::iterator it = m_FactoryMapLayout.find(layoutClassName);
    if (it != m_FactoryMapLayout.end())
        return it->second();
    return NULL;
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  simDoc  the TiXmlDocument to read from.
 *  @return true if successful, false otherwise.
 */
bool FClassOfCategory::readParameters(TiXmlDocument* simDoc)
{
    TiXmlElement* parms = NULL;

    if ((parms = simDoc->FirstChildElement()->FirstChildElement("ModelParams")) == NULL) {
        cerr << "Could not find <ModelParms> in simulation parameter file " << endl;
        return false;
    }

    try {
         parms->Accept(this);
    } catch (ParseParamError &error) {
        error.print(cerr);
        cerr << endl;
        return false;
    }

    // check to see if all required parameters were successfully read
    if (m_neurons->checkNumParameters() != true) {
        cerr << "Some parameters are missing in <NeuronsParams> in simulation parameter file " << endl;
        return false;
    }
    if (m_synapses->checkNumParameters() != true) {
        cerr << "Some parameters are missing in <SynapsesParams> in simulation parameter file " << endl;
        return false;
    }
    if (m_conns->checkNumParameters() != true) {
        cerr << "Some parameters are missing in <ConnectionsParams> in simulation parameter file " << endl;
        return false;
    }
    if (m_layout->checkNumParameters() != true) {
        cerr << "Some parameters are missing in <LayoutParams> in simulation parameter file " << endl;
        return false;
    }

    return true;
}

/*
 *  Read Parameters and parse an element for parameter values.
 *  Takes an XmlElement and checks for errors. If not, calls getValueList().
 *
 *  @param  element TiXmlElement to examine.
 *  @param  firstAttribute  ***NOT USED***.
 *  @return true if method finishes without errors.
 */
bool FClassOfCategory::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
//TODO: firstAttribute does not seem to be used! Delete?
{
    enum modelParams {neuronsParams = 1, synapsesParams = 2, connectionsParams = 3, layoutParams = 4, undefParams = 0};
    static modelParams paramsType;

    if (element.ValueStr().compare("ModelParams") == 0) {
        cerr << "Looking at ModelParams" << endl;
		paramsType = undefParams;
        return true;
    }
    if (element.ValueStr().compare("NeuronsParams") == 0) {
		cerr << "Looking at NeuronsParams" << endl;
        paramsType = neuronsParams;
        return true;
    }
    if (element.ValueStr().compare("SynapsesParams") == 0) {
		cerr << "Looking at SynapsesParams" << endl;
        paramsType = synapsesParams;
        return true;
    }
    if (element.ValueStr().compare("ConnectionsParams") == 0) {
		cerr << "Looking at ConnectionsParams" << endl;
        paramsType = connectionsParams;
        return true;
    }
    if (element.ValueStr().compare("LayoutParams") == 0) {
		cerr << "Looking at LayoutParams" << endl;
        paramsType = layoutParams;
        return true;
    }

    // Considering the duplication of element name between different model categories,
    // so we call readParameters separately based on the current parameters type.

    // Read neurons parameters
    if ((paramsType == neuronsParams) && (m_neurons->readParameters(element) != true)) {
        // If failed, we have unrecognized parameters.
        throw ParseParamError("FClassOfCategory", "Unrecognized neurons parameter '" + element.ValueStr() + "' was detected.");
    }
    // Read synapses parameters
    if ((paramsType == synapsesParams) && (m_synapses->readParameters(element) != true)) {
        // If failed, we have unrecognized parameters.
        throw ParseParamError("FClassOfCategory", "Unrecognized synapses parameter '" + element.ValueStr() + "' was detected.");
    } 
    // Read connections parameters
    if ((paramsType == connectionsParams) && (m_conns->readParameters(element) != true)) {
        // If failed, we have unrecognized parameters.
        throw ParseParamError("FClassOfCategory", "Unrecognized connections parameter '" + element.ValueStr() + "' was detected.");
    }
    // Read layout parameters
    if ((paramsType == layoutParams) && (m_layout->readParameters(element) != true)) { 
        cerr << "We end up throwing a layout error anyway" << endl;	
        // If failed, we have unrecognized parameters.
        throw ParseParamError("FClassOfCategory", "Unrecognized layout parameter '" + element.ValueStr() + "' was detected.");
    }

    return true;
}

/*
 *  Prints out all parameters of the model to ostream.
 *
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
