/**
 * Maintains intra-epoch state of connections in the network. This includes history and parameters
 * that inform how new connections are made during growth.
 */

#pragma once

#include "Global.h"
#include "Connections.h"
#include "SimulationInfo.h"
#include <vector>
#include <iostream>

using namespace std;

class ConnStatic : public Connections
{
    public:
        // TODO
        ConnStatic();
        virtual ~ConnStatic();

        static Connections* Create() { return new ConnStatic(); }

        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, AllNeurons *neurons, AllSynapses *synapses);
        virtual void cleanupConnections();
        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void readConns(istream& input, const SimulationInfo *sim_info);
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info);
        virtual IRecorder* createRecorder(const string &stateOutputFileName, IModel *model, const SimulationInfo *sim_info);

    private:
        // number of maximum connections per neurons
        int nConnsPerNeuron;
        // Connection radius threshold
        BGFLOAT threshConnsRadius;
        // Small-world rewiring probability
        BGFLOAT pRewiring;

        struct DistDestNeuron
        {
            BGFLOAT dist;
            int dest_neuron;

            bool operator<(const DistDestNeuron& other) const
            {
                return (dist < other.dist);
            }
        };
};
