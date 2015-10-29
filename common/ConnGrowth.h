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

class ConnGrowth : public Connections
{
    public:
        // TODO
        int *spikeCounts;

        //! synapse weight
        CompleteMatrix *W;
        //! neuron radii
        VectorMatrix *radii;
        //! spiking rate
        VectorMatrix *rates;
        //! distance between connection frontiers
        CompleteMatrix *delta;
        //! areas of overlap
        CompleteMatrix *area;
        //! neuron's outgrowth
        VectorMatrix *outgrowth;
        //! displacement of neuron radii
        VectorMatrix *deltaR;

        // TODO
        ConnGrowth();
        virtual ~ConnGrowth();

        static Connections* Create() { return new ConnGrowth(); }

        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, AllNeurons *neurons, AllSynapses *synapses);
        virtual void cleanupConnections();
        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void readConns(istream& input, const SimulationInfo *sim_info);
        virtual void writeConns(ostream& output, const SimulationInfo *sim_info);
        virtual bool updateConnections(AllNeurons &neurons, const SimulationInfo *sim_info, Layout *layout);
        virtual IRecorder* createRecorder(const string &stateOutputFileName, IModel *model, const SimulationInfo *sim_info);
#if defined(USE_GPU)
        virtual void updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeurons* m_allNeuronsDevice, AllSpikingSynapses* m_allSynapsesDevice, Layout *layout);
#else
        virtual void updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, Layout *layout);
#endif

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
        void updateConns(AllNeurons &neurons, const SimulationInfo *sim_info);
        void updateFrontiers(const int num_neurons, Layout *layout);
        void updateOverlap(BGFLOAT num_neurons, Layout *layout);
};

#if defined(__CUDACC__)
//! Update the network.
extern __global__ void updateSynapsesWeightsDevice( int num_neurons, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, void (*fpCreateSynapse)(AllSpikingSynapses*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType), neuronType* neuron_type_map_d );
#endif
