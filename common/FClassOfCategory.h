/**
 *      @file FClassOfCategory.h
 *
 *      @brief A factoy class for creating an instance of class of each category
 */

/**
 **
 ** @class FClassOfCategory FClassOfCategory.h "FClassOfCategory.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The FClassOfCategory creates an instance of the class specified by each category.
 ** Class categories are neurons, synapses, connections, and layout. 
 ** The factory reads the parameter file and look for the class name of each category.
 ** When the class name of the categry is found, an instance of the class will be created,
 ** and initialized by calling readParameters() method of the class.
 **
 ** The following is the step to add a new class: 1) Register the class name and the method
 ** to create an instance of the class in the factory; 2) Prepare the parameter file of the 
 ** class; 3) Export Create() method of the class.

 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#include <map>
#include "Global.h"
#include "IAllNeurons.h"
#include "IAllSynapses.h"
#include "Connections.h"
#include "Layout.h"

class FClassOfCategory : public TiXmlVisitor
{
public:
    //! The constructor for FClassOfCategory.
    FClassOfCategory();
    ~FClassOfCategory();

    static FClassOfCategory *get()
    {
        static FClassOfCategory instance;
        return &instance;
    }

    /**
     * Create an instance of the neurons class, which is specified in the parameter file.
     *
     * @param  element TiXmlNode to examine.
     * @return Poiner to the neurons object.
     */
    IAllNeurons* createNeurons(const TiXmlNode* parms);

    /**
     * Create an instance of the synapses class, which is specified in the parameter file.
     *
     * @param  element TiXmlNode to examine.
     * @return Poiner to the synapses object.
     */
    IAllSynapses* createSynapses(const TiXmlNode* parms);

    /**
     * Create an instance of the connections class, which is specified in the parameter file.
     *
     * @param  element TiXmlNode to examine.
     * @return Poiner to the connections object.
     */
    Connections* createConnections(const TiXmlNode* parms);

    /**
     * Create an instance of the layout class, which is specified in the parameter file.
     *
     * @param  element TiXmlNode to examine.
     * @return Poiner to the layout object.
     */
    Layout* createLayout(const TiXmlNode* parms);

    /**
     *  Attempts to read parameters from a XML file.
     *
     *  @param  simDoc TiXmlDocument to examine.
     *  @return true if successful, false otherwise.
     */
    bool readParameters(TiXmlDocument* simDoc);

    /**
     *  Prints out all parameters of the neurons to ostream.
     *
     *  @param  output  ostream to send output to.
     */
    void printParameters(ostream &output) const;

protected:
    /**
     *  Read Parameters and parse an element for parameter values.
     *  Takes an XmlElement and checks for errors. If not, calls getValueList().
     *
     *  @param  element TiXmlElement to examine.
     *  @param  firstAttribute  ***NOT USED***.
     *  @return true if method finishes without errors.
     */
     virtual bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);

private:
    //! Pointer to the neurons object.
    IAllNeurons* m_neurons;

    //! Pointer to the synapses objet.
    IAllSynapses* m_synapses;

    //! Pointer to the connections object.
    Connections* m_conns;

    //! Pointer to the layout object.
    Layout* m_layout;

    // type definitions
    typedef IAllNeurons* (*CreateNeuronsFn)(void);
    typedef IAllSynapses* (*CreateSynapsesFn)(void);
    typedef Connections* (*CreateConnsFn)(void);
    typedef Layout* (*CreateLayoutFn)(void);

    typedef map<string, CreateNeuronsFn> FactoryMapNeurons;
    typedef map<string, CreateSynapsesFn> FactoryMapSynapses;
    typedef map<string, CreateConnsFn> FactoryMapConns;
    typedef map<string, CreateLayoutFn> FactoryMapLayout;

    //! neurons class name, class creation function map
    FactoryMapNeurons m_FactoryMapNeurons;

    //! synapses class name, class creation function map
    FactoryMapSynapses m_FactoryMapSynapses;

    //! connections class name, class creation function map
    FactoryMapConns m_FactoryMapConns;

    //! layout class name, class creation function map
    FactoryMapLayout m_FactoryMapLayout;

    /**
     *  Register neurons class and its creation function to the factory.
     *
     *  @param  neuronsClassName  neurons class name.
     *  @param  pfnCreateNeurons  Pointer to the class creation function.
     */
    void registerNeurons(const string &neuronsClassName, CreateNeuronsFn pfnCreateNeurons);

    /**
     *  Register synapses class and its creation function to the factory.
     *
     *  @param  synapsesClassName synapses class name.
     *  @param  pfnCreateNeurons  Pointer to the class creation function.
     */
    void registerSynapses(const string &synapsesClassName, CreateSynapsesFn pfnCreateSynapses);

    /**
     *  Register connections class and its creation function to the factory.
     *
     *  @param  connsClassName    connections class name.
     *  @param  pfnCreateNeurons  Pointer to the class creation function.
     */
    void registerConns(const string &connsClassName, CreateConnsFn pfnCreateConns);

    /**
     *  Register layout class and its creation function to the factory.
     *  
     *  @param  layoutClassName   layout class name.
     *  @param  pfnCreateNeurons  Pointer to the class creation function.
     */
    void registerLayout(const string &layoutClassName, CreateLayoutFn pfnCreateLayout);

    /**
     * Create an instance of the neurons class, which name is specified by neuronsClassName.
     *
     * @param  neuronsClassName neurons class name to create.
     * @return Poiner to the neurons object.
     */
    IAllNeurons* createNeuronsWithName(const string& neuronsClassName);

    /**
     * Create an instance of the synapses class, which name is specified by synapsesClassName.
     *
     * @param  synapsesClassName synapses class name to create.
     * @return Poiner to the synapses object.
     */
    IAllSynapses* createSynapsesWithName(const string& synapsesClassName);

    /**
     * Create an instance of the connections class, which name is specified by connsClassName.
     *
     * @param  connsClassName connections class name to create.
     * @return Poiner to the connections object.
     */
    Connections* createConnsWithName(const string& connsClassName);

    /**
     * Create an instance of the layout class, which name is specified by layoutClassName.
     *
     * @param  layoutClassName layout class name to create.
     * @return Poiner to the layout object.
     */
    Layout* createLayoutWithName(const string& layoutClassName);
};

