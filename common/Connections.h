/**
 * Maintains intra-epoch state of connections in the network. This includes history and parameters
 * that inform how new connections are made during growth.
 */

#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "AllSpikingNeurons.h"
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
        int *spikeCounts;

        // TODO
        VectorMatrix *xloc;
        // TODO
        VectorMatrix *yloc;

        //! synapse weight
        CompleteMatrix *W;
        //! neuron radii
        VectorMatrix *radii;
        //! spiking rate
        VectorMatrix *rates;
        //! Inter-neuron distance squared
        CompleteMatrix *dist2;
        //! distance between connection frontiers
        CompleteMatrix *delta;
        //! the true inter-neuron distance
        CompleteMatrix *dist;
        //! areas of overlap
        CompleteMatrix *area;
        //! neuron's outgrowth
        VectorMatrix *outgrowth;
        //! displacement of neuron radii
        VectorMatrix *deltaR;

        // TODO
        Connections();
        virtual ~Connections();

        virtual void setupConnections(const SimulationInfo *sim_info);
        virtual void cleanupConnections();
        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void readConns(istream& input, const SimulationInfo *sim_info);
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info);
        virtual void updateConns(AllNeurons &neurons, const SimulationInfo *sim_info);
        virtual void updateFrontiers(const int num_neurons);
        virtual void updateOverlap(BGFLOAT num_neurons);

        struct GrowthParams
        {
            BGFLOAT epsilon; //null firing rate(zero outgrowth)
            BGFLOAT beta;  //sensitivity of outgrowth to firing rate
            BGFLOAT rho;  //outgrowth rate constant
            BGFLOAT targetRate; // Spikes/second
            BGFLOAT maxRate; // = targetRate / epsilon;
            BGFLOAT minRadius; // To ensure that even rapidly-firing neurons will connect to
                               // other neurons, when within their RFS.
            BGFLOAT startRadius; // No need to wait a long time before RFs start to overlap
        };

        // TODO
        GrowthParams m_growth;

    private:
};

