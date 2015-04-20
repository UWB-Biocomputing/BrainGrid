#pragma once
#include "GPUSpikingModel.h"
#include "AllIZHNeurons.h"

class IZHGPUModel : public GPUSpikingModel {

public:
    IZHGPUModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout);
    virtual ~IZHGPUModel();

    virtual void setupSim(SimulationInfo *sim_info, IRecorder* simRecorder);
    virtual void cleanupSim(SimulationInfo *sim_info);
    virtual void loadMemory(istream& input, const SimulationInfo *sim_info);

protected:
    virtual void advanceNeurons(const SimulationInfo *sim_info);

    virtual void copyDeviceSpikeHistoryToHost(AllSpikingNeurons &allNeuronsHost, const SimulationInfo *sim_info);
    virtual void copyDeviceSpikeCountsToHost(AllSpikingNeurons &allNeuronsHost, int numNeurons);
    virtual void clearSpikeCounts(int numNeurons);
    virtual void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);

private:

    /*----------------------------------------------*\
    |  Member variables
    \*----------------------------------------------*/

    //! Neuron structure in device memory.
    AllIZHNeurons* m_allNeuronsDevice;
};
