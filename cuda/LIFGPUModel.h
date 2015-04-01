#pragma once
#include "GPUSpikingModel.h"
#include "AllIFNeurons.h"

class LIFGPUModel : public GPUSpikingModel {

public:
    LIFGPUModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout);
    virtual ~LIFGPUModel();

    virtual void setupSim(SimulationInfo *sim_info, IRecorder* simRecorder);
    virtual void loadMemory(istream& input, const SimulationInfo *sim_info);
    virtual void cleanupSim(SimulationInfo *sim_info);

protected:
    virtual void advanceNeurons(const SimulationInfo *sim_info);
    virtual void advanceSynapses(const SimulationInfo *sim_info);
    virtual void calcSummationMap(const SimulationInfo *sim_info);

    virtual void copyDeviceSpikeHistoryToHost(AllSpikingNeurons &allNeuronsHost, const SimulationInfo *sim_info);
    virtual void copyDeviceSpikeCountsToHost(AllSpikingNeurons &allNeuronsHost, int numNeurons);
    virtual void clearSpikeCounts(int numNeurons);
    virtual void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);

private:
    /* ------------------*\
    |* # Helper Functions
    \* ------------------*/

    void copyDeviceSynapseCountsToHost(AllSynapses &allSynapsesHost, int neuron_count);
    void copyDeviceSynapseSumCoordToHost(AllSynapses &allSynapsesHost, int neuron_count, int max_synapses);

    //! Neuron structure in device memory.
    AllIFNeurons* m_allNeuronsDevice;
    
    //! Synapse structures in device memory.
    AllDSSynapses* m_allSynapsesDevice;
};
