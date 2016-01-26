/**
 *      @file Connections.h
 *
 *      @brief The base class of all connections classes
 */

/**
 *
 * @class Connections Connections.h "Connections.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * A placeholder to define connections of neunal networks.
 * In neunal networks, neurons are connected through synapses where messages are exchanged.
 * The strength of connections is characterized by synapse's weight. 
 * The connections classes define topologies, the way to connect neurons,  
 * and dynamics, the way to change connections as time elapses, of the networks. 
 * 
 * Connections can be either static or dynamic. The static connectons are ones where
 * connections are established at initialization and never change. 
 * The dynamic connections can be changed as the networks evolve, so in the dynamic networks
 * synapses will be created, deleted, or their weight will be modifed.  
 *
 * Connections classes may maintains intra-epoch state of connections in the network. 
 * This includes history and parameters that inform how new connections are made during growth.
 * Therefore, connections classes will have customized recorder classes, and provide
 * a function to craete the recorder class.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Layout.h"
#include "IRecorder.h"
#include <vector>
#include <iostream>

using namespace std;

class IModel;

class Connections
{
    public:
        // TODO
        Connections();
        virtual ~Connections();

        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses) = 0;
        virtual void cleanupConnections() = 0;
        virtual bool readParameters(const TiXmlElement& element) = 0;
        virtual void printParameters(ostream &output) const = 0;
        virtual void readConns(istream& input, const SimulationInfo *sim_info) = 0;
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info) = 0;
        virtual bool updateConnections(IAllNeurons &neurons, const SimulationInfo *sim_info, Layout *layout);
        virtual IRecorder* createRecorder(const string &stateOutputFileName, IModel *model, const SimulationInfo *sim_info) = 0;
#if defined(USE_GPU)
        virtual void updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeurons* m_allNeuronsDevice, AllSpikingSynapses* m_allSynapsesDevice, Layout *layout);
#else
        virtual void updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, Layout *layout);
#endif

    private:
};

