/**
 * Maintains intra-epoch state of connections in the network. This includes history and parameters
 * that inform how new connections are made during growth.
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

        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, AllNeurons *neurons, AllSynapses *synapses) = 0;
        virtual void cleanupConnections() = 0;
        virtual bool readParameters(const TiXmlElement& element) = 0;
        virtual void printParameters(ostream &output) const = 0;
        virtual void readConns(istream& input, const SimulationInfo *sim_info) = 0;
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info) = 0;
        virtual bool updateConnections(AllNeurons &neurons, const SimulationInfo *sim_info, Layout *layout);
        virtual IRecorder* createRecorder(const string &stateOutputFileName, IModel *model, const SimulationInfo *sim_info) = 0;
#if defined(USE_GPU)
        virtual void updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeurons* m_allNeuronsDevice, AllSpikingSynapses* m_allSynapsesDevice, Layout *layout);
#else
        virtual void updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, Layout *layout);
#endif

    private:
};

