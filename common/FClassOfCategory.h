/**
 ** \brief A factoy class for creating a class of the category.
 **
 ** \class FClassOfCategory FClassOfCategory.h "FClassOfCategory.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The FClassOfCategory creates an instance of a class specified by the category.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

/**
 ** \file FClassOfCategory.h
 **
 ** \brief Header file for FClassOfCategory.
 **/

#pragma once

#include <map>
#include "Global.h"
#include "AllNeurons.h"
#include "AllSynapses.h"
#include "Connections.h"
#include "Layout.h"

class FClassOfCategory : public TiXmlVisitor
{
public:
    //! The constructor for FSInput.
    FClassOfCategory();
    ~FClassOfCategory();

    static FClassOfCategory *get()
    {
        static FClassOfCategory instance;
        return &instance;
    }

    //! Create an instance.
    AllNeurons* createNeurons(TiXmlElement* parms);
    AllSynapses* createSynapses(TiXmlElement* parms);
    Connections* createConnections(TiXmlElement* parms);
    Layout* createLayout(TiXmlElement* parms);

    bool readParameters(TiXmlElement *source);
    void printParameters(ostream &output) const;

protected:
        // # Read Parameters
        // -----------------

        // Parse an element for parameter values.
        // Required by TiXmlVisitor, which is used by #readParameters
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);

private:
    AllNeurons* m_neurons;
    AllSynapses* m_synapses;
    Connections* m_conns;
    Layout* m_layout;

    typedef AllNeurons* (*CreateNeuronsFn)(void);
    typedef AllSynapses* (*CreateSynapsesFn)(void);
    typedef Connections* (*CreateConnsFn)(void);
    typedef Layout* (*CreateLayoutFn)(void);

    typedef map<string, CreateNeuronsFn> FactoryMapNeurons;
    typedef map<string, CreateSynapsesFn> FactoryMapSynapses;
    typedef map<string, CreateConnsFn> FactoryMapConns;
    typedef map<string, CreateLayoutFn> FactoryMapLayout;

    FactoryMapNeurons m_FactoryMapNeurons;
    FactoryMapSynapses m_FactoryMapSynapses;
    FactoryMapConns m_FactoryMapConns;
    FactoryMapLayout m_FactoryMapLayout;

    void registerNeurons(const string &neuronsClassName, CreateNeuronsFn pfnCreateNeurons);
    void registerSynapses(const string &synapsesClassName, CreateSynapsesFn pfnCreateSynapses);
    void registerConns(const string &connsClassName, CreateConnsFn pfnCreateConns);
    void registerLayout(const string &layoutClassName, CreateLayoutFn pfnCreateLayout);

    AllNeurons* createNeuronsWithName(const string& neuronsClassName);
    AllSynapses* createSynapsesWithName(const string& synapsesClassName);
    Connections* createConnsWithName(const string& connsClassName);
    Layout* createLayoutWithName(const string& layoutClassName);
};

