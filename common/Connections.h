/**
 * Maintains intra-epoch state of connections in the network. This includes history and parameters
 * that inform how new connections are made during growth.
 */

#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include <vector>
#include <iostream>

using namespace std;

class Connections
{
    public:
        // TODO
        static const string MATRIX_TYPE;
        // TODO
        static const string MATRIX_INIT;

        // TODO
        VectorMatrix *xloc;
        // TODO
        VectorMatrix *yloc;
        //! Inter-neuron distance squared
        CompleteMatrix *dist2;
        //! the true inter-neuron distance
        CompleteMatrix *dist;

        // TODO
        Connections();
        virtual ~Connections();

        virtual void setupConnections(const SimulationInfo *sim_info);
        virtual void cleanupConnections();
        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void readConns(istream& input, const SimulationInfo *sim_info);
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info);
        virtual bool updateConnections(AllNeurons &neurons, const SimulationInfo *sim_info);
#if defined(USE_GPU)
        virtual void updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeurons* m_allNeuronsDevice, AllSpikingSynapses* m_allSynapsesDevice) = 0;
#else
        virtual void updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);
#endif

    private:
};

